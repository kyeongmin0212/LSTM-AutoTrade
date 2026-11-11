"""업비트 시세로 LSTM이 다음 날 종가를 예측해 매매하고, 텔레그램으로 알리는 자동매매 봇.

공개용 기본 구조입니다. 실제로 사용한 종목 선정 기준과 매수 판단 로직은 포함하지 않았습니다.
"""
import os
import sys
import json
import time
import asyncio
import logging
import threading
import hashlib
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from dotenv import load_dotenv
import pyupbit
import ta
import telegram
from telegram import Bot
from filelock import FileLock
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator

# Windows 콘솔에서 유니코드 지원
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 상수 정의
COINS_FILE = "purchased_coins.json"
TRADE_HISTORY_FILE = "trade_history.json"
POSITIONS_FILE = "positions.json"
CACHE_DIR = "api_cache"
MIN_KRW_BALANCE = 5000
MAX_COINS = 5
API_CALL_LIMIT = 30
ORDER_LIMIT = 10
CACHE_EXPIRY_SECONDS = {
    'default': 3600,
    'market_trend': 7200,
    'price_prediction': 7200,
    'opportunities': 7200
}

# 캐시 디렉토리 생성
os.makedirs(CACHE_DIR, exist_ok=True)

# API 키 설정
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

logger.info("API 키 확인 중...")
missing_keys = []
for key, value in [
    ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
    ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
    ("UPBIT_ACCESS_KEY", UPBIT_ACCESS_KEY),
    ("UPBIT_SECRET_KEY", UPBIT_SECRET_KEY)
]:
    if not value:
        missing_keys.append(key)
if missing_keys:
    logger.error(f"누락된 API 키: {', '.join(missing_keys)}")
    raise ValueError("환경 변수 파일(.env)에 하나 이상의 API 키가 누락되었습니다.")

# 업비트 클라이언트 초기화
upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
logger.info("업비트 클라이언트 초기화 완료")

# JSON 직렬화를 위한 유틸리티 함수
def serialize_numpy(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Rate Limiter
class RateLimiter:
    def __init__(self):
        self.order_lock = threading.Lock()
        self.api_lock = threading.Lock()
        self.order_count = 0
        self.api_count = 0
        self.last_reset = time.time()

    def _reset_if_needed(self):
        current_time = time.time()
        if current_time - self.last_reset >= 1:
            with self.order_lock:
                self.order_count = 0
            with self.api_lock:
                self.api_count = 0
            self.last_reset = current_time

    def can_make_order(self):
        self._reset_if_needed()
        with self.order_lock:
            if self.order_count < ORDER_LIMIT:
                self.order_count += 1
                return True
            return False

    def can_make_api_call(self, max_wait=10):
        self._reset_if_needed()
        start_time = time.time()
        with self.api_lock:
            while self.api_count >= API_CALL_LIMIT:
                if time.time() - start_time > max_wait:
                    logger.warning("API 호출 대기 시간 초과")
                    return False
                time.sleep(0.1)
                self._reset_if_needed()
            self.api_count += 1
            return True

rate_limiter = RateLimiter()

# 캐시 데코레이터
def cache_result(expiry_seconds=14400):
    def decorator(func):
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        async def async_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
            lock_file = f"{cache_file}.lock"

            with FileLock(lock_file):
                try:
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                            if time.time() - cache_data.get('timestamp', 0) < expiry_seconds:
                                logger.info(f"{func.__name__} 캐시 결과 사용")
                                return cache_data['result']
                except Exception as e:
                    logger.error(f"{func.__name__} 캐시 읽기 실패: {e}")

                result = await func(*args, **kwargs)

                try:
                    cache_data = {
                        'timestamp': time.time(),
                        'result': json.loads(json.dumps(result, default=serialize_numpy))
                    }
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False)
                    logger.info(f"{func.__name__} 캐시 저장 완료")
                except Exception as e:
                    logger.error(f"{func.__name__} 캐시 저장 실패: {e}")

                return result

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
            lock_file = f"{cache_file}.lock"

            with FileLock(lock_file):
                try:
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                            if time.time() - cache_data.get('timestamp', 0) < expiry_seconds:
                                logger.info(f"{func.__name__} 캐시 결과 사용")
                                return cache_data['result']
                except Exception as e:
                    logger.error(f"{func.__name__} 캐시 읽기 실패: {e}")

                result = func(*args, **kwargs)

                try:
                    cache_data = {
                        'timestamp': time.time(),
                        'result': json.loads(json.dumps(result, default=serialize_numpy))
                    }
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False)
                    logger.info(f"{func.__name__} 캐시 저장 완료")
                except Exception as e:
                    logger.error(f"{func.__name__} 캐시 저장 실패: {e}")

                return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def invalidate_cache(func_name, ticker):
    cache_key = f"{func_name}_{hashlib.md5(ticker.encode()).hexdigest()}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        logger.info(f"{func_name} 캐시 무효화: {ticker}")

class TradingError(Exception):
    pass

class FileManager:
    @staticmethod
    def load_json(filename, default=None):
        lock_file = f"{filename}.lock"
        with FileLock(lock_file):
            try:
                if os.path.exists(filename):
                    with open(filename, "r", encoding='utf-8') as f:
                        return json.load(f)
                return default if default is not None else {}
            except Exception as e:
                logger.error(f"{filename} 파일 로드 실패: {e}")
                return default if default is not None else {}

    @staticmethod
    def save_json(filename, data):
        lock_file = f"{filename}.lock"
        with FileLock(lock_file):
            try:
                with open(filename, "w", encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x.tolist() if isinstance(x, np.ndarray) else str(x))
                logger.info(f"{filename} 파일 저장 완료")
            except Exception as e:
                logger.error(f"{filename} 파일 저장 실패: {e}")

async def async_send_telegram_message(message, max_retries=3):
    for attempt in range(max_retries):
        try:
            max_length = 4096
            if len(message) > max_length:
                message = message[:max_length - 3] + "..."
            logger.info(f"텔레그램 메시지 전송 시도 {attempt + 1}/{max_retries}: {message[:100]}...")
            if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
                logger.error("텔레그램 토큰 또는 채팅 ID 누락")
                return False
            bot = Bot(token=TELEGRAM_TOKEN)
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            logger.info("텔레그램 메시지 전송 성공")
            return True
        except telegram.error.TelegramError as e:
            logger.error(f"텔레그램 메시지 전송 실패: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"텔레그램 메시지 전송 중 예상치 못한 오류: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    logger.error("텔레그램 메시지 전송 실패: 최대 재시도 횟수 초과")
    return False

class TradingHistory:
    def __init__(self):
        self.history = FileManager.load_json(TRADE_HISTORY_FILE, {"trades": [], "performance": {}, "blacklist": {}})
        logger.info("거래 내역 초기화 완료")

    def add_to_blacklist(self, ticker, cooldown_hours=4):
        """손절 후 코인을 일정 시간 블랙리스트에 추가"""
        expiry = (datetime.now() + timedelta(hours=cooldown_hours)).isoformat()
        self.history["blacklist"][ticker] = expiry
        FileManager.save_json(TRADE_HISTORY_FILE, self.history)
        logger.info(f"{ticker} 블랙리스트 추가: {cooldown_hours}시간")

    def is_blacklisted(self, ticker):
        blacklist = self.history.get("blacklist", {})  # 안전하게 접근
        logger.info(f"블랙리스트 확인: {blacklist}")
        if ticker in blacklist:
            expiry = datetime.fromisoformat(blacklist[ticker])
            if datetime.now() < expiry:
                logger.info(f"{ticker} 블랙리스트에 있음, 만료: {expiry}")
                return True
            else:
                del blacklist[ticker]
                FileManager.save_json(TRADE_HISTORY_FILE, self.history)
                logger.info(f"{ticker} 블랙리스트 만료")
        return False
    
    async def add_trade(self, ticker, action, price, amount, reason):
        trade = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "action": action,
            "price": float(price),
            "amount": float(amount),
            "reason": reason
        }
        self.history["trades"].append(trade)
        FileManager.save_json(TRADE_HISTORY_FILE, self.history)
        action_text = "매수" if action == "buy" else "매도"
        message = (
            f"🔄 거래 실행: {ticker}\n"
            f"유형: {action_text}\n"
            f"가격: {price:,.2f}원\n"
            f"수량: {amount:.8f}개\n"
            f"사유: {reason}"
        )
        await async_send_telegram_message(message)
        logger.info(f"{ticker} {action_text} 거래 기록 추가: 가격={price:,.2f}원, 수량={amount:.8f}개")

    def calculate_performance(self, ticker):
        if not rate_limiter.can_make_api_call():
            time.sleep(0.1)
            return self.calculate_performance(ticker)
        try:
            trades = [t for t in self.history["trades"] if t["ticker"] == ticker]
            if not trades:
                logger.info(f"{ticker} 거래 내역 없음")
                return None
            total_invested = sum(t["amount"] * t["price"] for t in trades if t["action"] == "buy")
            total_returned = sum(t["amount"] * t["price"] for t in trades if t["action"] == "sell")
            current_price = pyupbit.get_current_price(ticker)
            if current_price is None:
                logger.error(f"{ticker} 현재 가격 가져오기 실패")
                return None
            current_holdings = sum(t["amount"] for t in trades if t["action"] == "buy") - \
                             sum(t["amount"] for t in trades if t["action"] == "sell")
            current_value = current_holdings * current_price if current_holdings > 0 else 0
            performance = {
                "total_invested": float(total_invested),
                "total_returned": float(total_returned),
                "current_holdings": float(current_holdings),
                "current_value": float(current_value),
                "total_value": float(total_returned + current_value),
                "roi": float(((total_returned + current_value - total_invested) / total_invested * 100)) \
                    if total_invested > 0 else 0
            }
            logger.info(f"{ticker} 성과 계산: ROI={performance['roi']:.2f}%")
            return performance
        except Exception as e:
            logger.error(f"{ticker} 성과 계산 실패: {e}")
            return None

class StopLossManager:
    def __init__(self, default_stop_loss=5, default_take_profit=10):
        self.default_stop_loss = default_stop_loss
        self.default_take_profit = default_take_profit
        self.positions = FileManager.load_json(POSITIONS_FILE, {})
        logger.info("손절/익절 관리자 초기화 완료")

    def add_position(self, ticker, entry_price, amount, stop_loss=None, take_profit=None):
        position = {
            "entry_price": float(entry_price),
            "amount": float(amount),
            "stop_loss": float(stop_loss or self.default_stop_loss),
            "take_profit": float(take_profit or self.default_take_profit),
            "timestamp": datetime.now().isoformat(),
            "stop_loss_order_uuid": None,
            "take_profit_order_uuid": None
        }
        self.positions[ticker] = position
        FileManager.save_json(POSITIONS_FILE, self.positions)
        logger.info(f"{ticker} 포지션 추가: 진입가={entry_price:,.2f}원, 수량={amount:.8f}개")
        self._place_limit_orders(ticker, entry_price, amount, position)

    def _place_limit_orders(self, ticker, entry_price, amount, position):
        try:
            stop_loss_price = entry_price * (1 - position["stop_loss"] / 100)
            take_profit_price = entry_price * (1 + position["take_profit"] / 100)
            if rate_limiter.can_make_order():
                sl_order = upbit.sell_limit_order(ticker, stop_loss_price, amount)
                if sl_order and "uuid" in sl_order:
                    position["stop_loss_order_uuid"] = sl_order["uuid"]
                    logger.info(f"{ticker} 손절 주문 설정: {stop_loss_price:,.2f}원")
            if rate_limiter.can_make_order():
                tp_order = upbit.sell_limit_order(ticker, take_profit_price, amount)
                if tp_order and "uuid" in tp_order:
                    position["take_profit_order_uuid"] = tp_order["uuid"]
                    logger.info(f"{ticker} 익절 주문 설정: {take_profit_price:,.2f}원")
            FileManager.save_json(POSITIONS_FILE, self.positions)
        except Exception as e:
            logger.error(f"{ticker} 지정가 주문 설정 실패: {e}")

    def update_position(self, ticker, new_avg_price=None, amount=None, stop_loss=None, take_profit=None):
        if ticker in self.positions:
            position = self.positions[ticker]
            if new_avg_price is not None:
                position["entry_price"] = float(new_avg_price)
            if amount is not None:
                position["amount"] = float(amount)
            if stop_loss is not None:
                position["stop_loss"] = float(stop_loss)
            if take_profit is not None:
                position["take_profit"] = float(take_profit)
            position["updated_at"] = datetime.now().isoformat()
            for uuid_key in ["stop_loss_order_uuid", "take_profit_order_uuid"]:
                if position.get(uuid_key):
                    try:
                        upbit.cancel_order(position[uuid_key])
                        logger.info(f"{ticker} 기존 {uuid_key} 주문 취소")
                    except Exception as e:
                        logger.error(f"{ticker} 주문 취소 실패: {e}")
            self._place_limit_orders(ticker, position["entry_price"], position["amount"], position)
            FileManager.save_json(POSITIONS_FILE, self.positions)
            logger.info(f"{ticker} 포지션 업데이트: 평균가={new_avg_price:,.2f}원, 수량={amount:.8f}개")

    def remove_position(self, ticker):
        if ticker in self.positions:
            position = self.positions[ticker]
            for uuid_key in ["stop_loss_order_uuid", "take_profit_order_uuid"]:
                if position.get(uuid_key):
                    try:
                        upbit.cancel_order(position[uuid_key])
                        logger.info(f"{ticker} {uuid_key} 주문 취소")
                    except Exception as e:
                        logger.error(f"{ticker} 주문 취소 실패: {e}")
            del self.positions[ticker]
            FileManager.save_json(POSITIONS_FILE, self.positions)
            logger.info(f"{ticker} 포지션 제거")

    async def check_limit_orders(self):
        for ticker, position in list(self.positions.items()):
            try:
                for order_type, uuid_key in [("stop_loss", "stop_loss_order_uuid"), ("take_profit", "take_profit_order_uuid")]:
                    uuid = position.get(uuid_key)
                    if uuid and rate_limiter.can_make_api_call():
                        order_info = upbit.get_order(uuid)
                        if order_info and order_info["state"] == "done":
                            executed_price = float(order_info["price"])
                            executed_amount = float(order_info["executed_volume"])
                            reason = f"{order_type} 실행"
                            await TradingHistory().add_trade(ticker, "sell", executed_price, executed_amount, reason)
                            self.remove_position(ticker)
                            bot = TradingBot()
                            if ticker in bot.purchased_coins["coins"]:
                                bot.purchased_coins["coins"].remove(ticker)
                                FileManager.save_json(COINS_FILE, bot.purchased_coins)
                            if order_type == "stop_loss":
                                bot.trading_history.add_to_blacklist(ticker)  # 블랙리스트 추가
                                invalidate_cache("scan_for_opportunities", ticker)  # 캐시 무효화
                            message = (
                                f"{'🔴 손절' if order_type == 'stop_loss' else '🟢 익절'} 실행: {ticker}\n"
                                f"체결가: {executed_price:,.2f}원\n"
                                f"수량: {executed_amount:.8f}개\n"
                                f"총 금액: {executed_amount * executed_price:,.0f}원"
                            )
                            await async_send_telegram_message(message)
            except Exception as e:
                logger.error(f"{ticker} 지정가 주문 확인 실패: {e}")

class MarketAnalyzer:
    def __init__(self):
        os.makedirs("lstm_models", exist_ok=True)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_lstm_model()
        logger.info("시장 분석기 초기화 완료")

    def _build_lstm_model(self):
        model = Sequential([
            Input(shape=(60, 1)),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(100),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        logger.info("LSTM 모델 생성 완료")
        return model

    @cache_result(expiry_seconds=CACHE_EXPIRY_SECONDS['price_prediction'])
    def predict_next_price(self, ticker):
        if not rate_limiter.can_make_api_call():
            time.sleep(0.1)
            return self.predict_next_price(ticker)
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=180)
            if df is None or len(df) < 60:
                logger.warning(f"{ticker} 데이터 부족: {len(df) if df is not None else 0}일")
                return None
            close_prices = df['close'].values.reshape(-1, 1)
            scaled_close = self.scaler.fit_transform(close_prices)
            X = []
            for i in range(60, len(scaled_close)):
                X.append(scaled_close[i-60:i])
            X = np.array(X)
            if len(X) == 0:
                logger.warning(f"{ticker} 유효한 시퀀스 없음")
                return None
            y = scaled_close[60:]
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            model_file = f"lstm_models/lstm_model_{ticker}.h5"
            if os.path.exists(model_file):
                self.model = tf.keras.models.load_model(model_file)
                logger.info(f"{ticker} 기존 모델 로드")
            history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)
            logger.info(f"{ticker} LSTM 학습 손실: {history.history['loss'][-1]:.6f}, 검증 손실: {history.history['val_loss'][-1]:.6f}, "
                        f"손실 추이: {[f'{x:.6f}' for x in history.history['loss']]}")
            self.model.save(model_file)
            logger.info(f"{ticker} 모델 저장: {model_file}")
            last_60_days = scaled_close[-60:].reshape((1, 60, 1))
            predicted_scaled = self.model.predict(last_60_days, verbose=0)
            predicted_price = float(self.scaler.inverse_transform(predicted_scaled)[0][0])
            actual_price = df['close'].iloc[-1]
            prediction_error = abs(predicted_price - actual_price) / actual_price * 100
            logger.info(f"{ticker} 다음 가격 예측: {predicted_price:,.2f}원, 실제가: {actual_price:,.2f}원, 예측 오차: {prediction_error:.2f}%")
            return predicted_price
        except Exception as e:
            logger.error(f"{ticker} 가격 예측 실패: {e}")
            return None

    def add_technical_indicators(self, df):
        try:
            df = df.copy()
            min_length = 26
            if len(df) < min_length:
                logger.warning(f"데이터 길이 {len(df)}가 최소 요구 길이 {min_length}보다 짧음")
                return df
            indicator_bb = BollingerBands(close=df['close'])
            df['bb_bbm'] = indicator_bb.bollinger_mavg()
            df['bb_bbh'] = indicator_bb.bollinger_hband()
            df['bb_bbl'] = indicator_bb.bollinger_lband()
            df['rsi'] = RSIIndicator(close=df['close']).rsi()
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
            df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
            logger.info("기술적 지표 추가 완료")
            return df
        except Exception as e:
            logger.error(f"기술적 지표 추가 실패: {e}")
            return df

    async def scan_for_opportunities(self, limit=5):
        """평가할 종목 후보를 고른다.

        공개용 예시: 실제로 사용한 종목 선정 기준은 포함하지 않았고,
        원화 마켓 종목을 앞에서부터 limit개 고른다.
        """
        try:
            tickers = [t for t in pyupbit.get_tickers(fiat="KRW") if t.startswith("KRW-")]
        except Exception as e:
            logger.error(f"티커 가져오기 실패: {e}")
            return []
        return [{"ticker": t, "reason": "후보 종목"} for t in tickers[:limit]]

    def test_lstm_prediction(self, ticker):
        predicted_price = self.predict_next_price(ticker)
        if predicted_price:
            logger.info(f"{ticker} 테스트 예측 완료: {predicted_price:,.2f}원")
            return predicted_price
        return None

    def monitor_prediction_accuracy(self, ticker):
        df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
        if df is None or len(df) < 2:
            logger.warning(f"{ticker} 모니터링 데이터 부족")
            return
        actual_price = df['close'].iloc[-1]
        predicted_price = self.predict_next_price(ticker)
        if predicted_price:
            error = abs(predicted_price - actual_price) / actual_price * 100
            logger.info(f"{ticker} 예측 정확도: 실제가={actual_price:,.2f}원, 예측가={predicted_price:,.2f}원, 오차={error:.2f}%")

    def backtest_lstm(self, ticker, days=30):
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=days + 60)
            if df is None or len(df) < 60:
                logger.warning(f"{ticker} 백테스팅 데이터 부족: {len(df) if df is not None else 0}일")
                return None
            errors = []
            for i in range(60, len(df) - 1):
                temp_df = df.iloc[:i]
                close_prices = temp_df['close'].values.reshape(-1, 1)
                scaled_close = self.scaler.fit_transform(close_prices)
                X = [scaled_close[j-60:j] for j in range(60, len(scaled_close))]
                X = np.array(X)
                y = scaled_close[60:]
                self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                last_60_days = scaled_close[-60:].reshape((1, 60, 1))
                predicted_scaled = self.model.predict(last_60_days, verbose=0)
                predicted_price = self.scaler.inverse_transform(predicted_scaled)[0][0]
                actual_price = df['close'].iloc[i]
                error = abs(predicted_price - actual_price) / actual_price * 100
                errors.append(error)
                logger.info(f"{ticker} 백테스팅 [{i-59}/{len(df)-60}]: 예측가={predicted_price:,.2f}원, 실제가={actual_price:,.2f}원, 오차={error:.2f}%")
            avg_error = sum(errors) / len(errors) if errors else 0
            logger.info(f"{ticker} 백테스팅 완료: 평균 예측 오차={avg_error:.2f}%")
            return avg_error
        except Exception as e:
            logger.error(f"{ticker} 백테스팅 실패: {e}")
            return None

    def retrain_on_loss(self, ticker, trade):
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=180)
            if df is None or len(df) < 60:
                logger.warning(f"{ticker} 재학습 데이터 부족")
                return
            close_prices = df['close'].values.reshape(-1, 1)
            scaled_close = self.scaler.fit_transform(close_prices)
            X = [scaled_close[i-60:i] for i in range(60, len(scaled_close))]
            X = np.array(X)
            y = scaled_close[60:]
            trade_price = trade['price']
            trade_time = datetime.fromisoformat(trade['timestamp'])
            df_index = df.index.get_loc(trade_time, method='nearest')
            if df_index >= 60:
                X_new = scaled_close[df_index-60:df_index].reshape((1, 60, 1))
                y_new = scaled_close[df_index]
                self.model.fit(X_new, np.array([y_new]), epochs=5, verbose=0)
                logger.info(f"{ticker} 손실 데이터로 재학습 완료")
                self.model.save(f"lstm_models/lstm_model_{ticker}.h5")
        except Exception as e:
            logger.error(f"{ticker} 손실 재학습 실패: {e}")

class TradingBot:
    def __init__(self):
        self.trading_history = TradingHistory()
        self.stop_loss_manager = StopLossManager()
        self.market_analyzer = MarketAnalyzer()
        self.purchased_coins = FileManager.load_json(COINS_FILE, {"coins": [], "last_analysis": {}})
        self.last_opportunity_scan = 0
        self.opportunities = []
        self.last_portfolio_error_time = 0
        self.sync_with_upbit_account()
        logger.info("트레이딩 봇 초기화 완료")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def sync_with_upbit_account(self):
        try:
            balances = upbit.get_balances()
            if not balances:
                logger.error("Upbit 계정 정보를 가져오지 못했습니다.")
                return
            current_coins = set()
            for balance in balances:
                currency = balance['currency']
                unit_currency = balance.get('unit_currency', 'KRW')
                ticker = f"{unit_currency}-{currency}"
                amount = float(balance['balance'])
                if amount > 0 and ticker.startswith('KRW-') and ticker != 'KRW-KRW':
                    current_coins.add(ticker)
                    if ticker not in self.purchased_coins["coins"]:
                        self.purchased_coins["coins"].append(ticker)
                        logger.info(f"{ticker} 코인을 Upbit 계정에서 발견하여 추가: {amount:.8f}개")
                    if ticker not in self.stop_loss_manager.positions:
                        avg_price = float(balance.get('avg_buy_price', 0))
                        if avg_price == 0:
                            current_price = pyupbit.get_current_price(ticker)
                            avg_price = current_price if current_price else 0
                        self.stop_loss_manager.add_position(ticker, avg_price, amount)
                        logger.info(f"{ticker} 포지션 추가: 매수가={avg_price:,.2f}원, 수량={amount:.8f}개")
            for ticker in self.purchased_coins["coins"][:]:
                if ticker not in current_coins:
                    self.purchased_coins["coins"].remove(ticker)
                    self.stop_loss_manager.remove_position(ticker)
                    logger.info(f"{ticker} 코인을 Upbit 계정에서 제거")
            FileManager.save_json(COINS_FILE, self.purchased_coins)
            FileManager.save_json(POSITIONS_FILE, self.stop_loss_manager.positions)
            logger.info("Upbit 계정 동기화 완료")
        except Exception as e:
            logger.error(f"Upbit 계정 동기화 실패: {e}", exc_info=True)
            raise

    async def find_trading_opportunities(self):
        self.opportunities = await self.market_analyzer.scan_for_opportunities()
        logger.info(f"후보 종목 {len(self.opportunities)}개 선정")

    async def execute_trading_strategy(self):
        """매수 판단.

        공개용 예시: LSTM 예측 종가가 현재가보다 높으면 매수한다.
        실제로 사용한 매수 조건은 포함하지 않았다.
        """
        try:
            if len(self.purchased_coins["coins"]) >= MAX_COINS:
                logger.info(f"최대 {MAX_COINS}개 코인 보유 중, 매수 건너뜀")
                return
            krw_balance = upbit.get_balance("KRW")
            if krw_balance is None or krw_balance < MIN_KRW_BALANCE:
                logger.info(f"KRW 잔고 부족: {krw_balance if krw_balance else '없음'}원")
                return
            for opp in self.opportunities:
                ticker = opp["ticker"]
                if ticker in self.purchased_coins["coins"] or self.trading_history.is_blacklisted(ticker):
                    continue
                predicted_price = self.market_analyzer.predict_next_price(ticker)
                current_price = pyupbit.get_current_price(ticker)
                if not predicted_price or not current_price or predicted_price <= current_price:
                    logger.info(f"{ticker} 매수 조건 미충족: 예측가={predicted_price or '없음'}, 현재가={current_price or '없음'}")
                    continue
                reason = f"LSTM 예측가 {predicted_price:,.2f}원 > 현재가 {current_price:,.2f}원"
                if await self.execute_buy(ticker, reason, budget=krw_balance / MAX_COINS):
                    break
        except Exception as e:
            logger.error(f"매수 전략 실행 실패: {e}", exc_info=True)

    async def _execute_stop_loss(self, ticker, position, current_price):
        if not rate_limiter.can_make_order():
            await asyncio.sleep(0.1)
            return await self._execute_stop_loss(ticker, position, current_price)
        try:
            amount = position["amount"]
            sell_result = upbit.sell_market_order(ticker, amount)
            if sell_result and "uuid" in sell_result:
                await self.trading_history.add_trade(
                    ticker, "sell", current_price, amount,
                    f"손절 (손실: {((current_price - position['entry_price']) / position['entry_price'] * 100):.2f}%)"
                )
                self.stop_loss_manager.remove_position(ticker)
                if ticker in self.purchased_coins["coins"]:
                    self.purchased_coins["coins"].remove(ticker)
                    FileManager.save_json(COINS_FILE, self.purchased_coins)
                self.trading_history.add_to_blacklist(ticker)  # 블랙리스트 추가
                invalidate_cache("scan_for_opportunities", ticker)  # 캐시 무효화
                message = (
                    f"🔴 손절 실행: {ticker}\n"
                    f"진입가: {position['entry_price']:,.2f}원\n"
                    f"청산가: {current_price:,.2f}원\n"
                    f"손실률: {((current_price - position['entry_price']) / position['entry_price'] * 100):.2f}%\n"
                    f"거래 금액: {amount * current_price:,.0f}원"
                )
                await async_send_telegram_message(message)
                logger.info(f"{ticker} 손절 실행 완료")
                self.sync_with_upbit_account()
                return True  # 성공 시 True 반환
            else:
                logger.error(f"{ticker} 손절 주문 실패: {sell_result or '응답 없음'}")
                return False  # 실패 시 False 반환
        except Exception as e:
            logger.error(f"{ticker} 손절 실행 실패: {e}")
            return False  # 예외 발생 시 False 반환        

    async def check_pending_orders(self):
        try:
            for ticker, position in list(self.stop_loss_manager.positions.items()):
                if position.get('status') != 'pending' or 'order_uuid' not in position:
                    continue
                order_uuid = position['order_uuid']
                order_info = upbit.get_order(order_uuid)
                if not order_info:
                    logger.warning(f"{ticker} 주문 정보 가져오기 실패, UUID: {order_uuid}")
                    continue
                if order_info['state'] == 'done':
                    logger.info(f"{ticker} 주문 체결됨")
                    executed_amount = float(order_info['executed_volume'])
                    avg_price = float(order_info['price'])
                    reason = position.get('reason', '매수')
                    await self.trading_history.add_trade(ticker, "buy", avg_price, executed_amount, reason)
                    position['status'] = 'active'
                    position['entry_price'] = avg_price
                    position['amount'] = executed_amount
                    self.stop_loss_manager.positions[ticker] = position
                    FileManager.save_json(POSITIONS_FILE, self.stop_loss_manager.positions)
                    if ticker not in self.purchased_coins["coins"]:
                        self.purchased_coins["coins"].append(ticker)
                        FileManager.save_json(COINS_FILE, self.purchased_coins)
                    message = (
                        f"✅ 매수 주문 체결: {ticker}\n"
                        f"체결가: {avg_price:,.2f}원\n"
                        f"수량: {executed_amount:.8f}개\n"
                        f"총 금액: {executed_amount * avg_price:,.0f}원\n"
                    )
                    await async_send_telegram_message(message)
                    self.sync_with_upbit_account()
                elif order_info['state'] == 'wait':
                    logger.info(f"{ticker} 주문 대기 중")
                elif order_info['state'] == 'cancel':
                    logger.info(f"{ticker} 주문 취소됨")
                    del self.stop_loss_manager.positions[ticker]
                    FileManager.save_json(POSITIONS_FILE, self.stop_loss_manager.positions)
        except Exception as e:
            logger.error(f"대기 주문 확인 오류: {e}")

    async def check_positions(self):
        tasks = []
        for ticker, position in list(self.stop_loss_manager.positions.items()):
            if not rate_limiter.can_make_api_call():
                await asyncio.sleep(0.1)
                continue
            tasks.append(self._check_position(ticker, position))
        await asyncio.gather(*tasks)

    async def _check_position(self, ticker, position):
        try:
            current_price = pyupbit.get_current_price(ticker)
            if not current_price:
                logger.error(f"{ticker} 현재 가격 가져오기 실패")
                return
            entry_price = position["entry_price"]
            profit_percent = ((current_price - entry_price) / entry_price) * 100
            if profit_percent <= -position["stop_loss"] and not position.get("stop_loss_order_uuid"):
                await self._execute_stop_loss(ticker, position, current_price)
            elif profit_percent >= position["take_profit"] and not position.get("take_profit_order_uuid"):
                await self._execute_take_profit(ticker, position, current_price)
        except Exception as e:
            logger.error(f"{ticker} 포지션 확인 실패: {e}")

    async def _execute_take_profit(self, ticker, position, current_price):
        if not rate_limiter.can_make_order():
            await asyncio.sleep(0.1)
            return await self._execute_take_profit(ticker, position, current_price)
        try:
            amount = position["amount"]
            sell_result = upbit.sell_market_order(ticker, amount)
            if sell_result and "uuid" in sell_result:
                await self.trading_history.add_trade(
                    ticker, "sell", current_price, amount,
                    f"익절 (수익: {((current_price - position['entry_price']) / position['entry_price'] * 100):.2f}%)"
                )
                self.stop_loss_manager.remove_position(ticker)
                if ticker in self.purchased_coins["coins"]:
                    self.purchased_coins["coins"].remove(ticker)
                    FileManager.save_json(COINS_FILE, self.purchased_coins)
                message = (
                    f"🟢 익절 실행: {ticker}\n"
                    f"진입가: {position['entry_price']:,.2f}원\n"
                    f"청산가: {current_price:,.2f}원\n"
                    f"수익률: {((current_price - position['entry_price']) / position['entry_price'] * 100):.2f}%\n"
                    f"거래 금액: {amount * current_price:,.0f}원"
                )
                await async_send_telegram_message(message)
                logger.info(f"{ticker} 익절 실행 완료")
                self.sync_with_upbit_account()
            else:
                logger.error(f"{ticker} 익절 주문 실패: {sell_result or '응답 없음'}")
        except Exception as e:
            logger.error(f"{ticker} 익절 실행 실패: {e}")

    async def execute_buy(self, ticker, reason, budget=None):
        if ticker in self.purchased_coins["coins"]:
            logger.info(f"{ticker} 이미 보유 중, 건너뜀")
            return False
        if not rate_limiter.can_make_order():
            await asyncio.sleep(0.1)
            return await self.execute_buy(ticker, reason, budget)
        try:
            current_price = pyupbit.get_current_price(ticker)
            if not current_price:
                logger.error(f"{ticker} 현재 가격 가져오기 실패")
                return False
            if budget is None:
                krw_balance = upbit.get_balance("KRW")
                if krw_balance is None:
                    logger.error("KRW 잔고 가져오기 실패, API 키 또는 네트워크 확인")
                    return False
                budget = krw_balance / MAX_COINS
            if budget < MIN_KRW_BALANCE:
                logger.error(f"예산 부족: {budget:,.2f}원 < {MIN_KRW_BALANCE}원")
                return False
            amount = budget / current_price
            buy_result = upbit.buy_market_order(ticker, budget)
            if buy_result and "uuid" in buy_result:
                order_uuid = buy_result["uuid"]
                order_info = upbit.get_order(order_uuid)
                for _ in range(10):
                    if order_info and order_info["state"] in ["done", "cancel"]:
                        break
                    await asyncio.sleep(1)
                    order_info = upbit.get_order(order_uuid)
                executed_amount = float(order_info.get("executed_volume", 0)) if order_info else 0
                if executed_amount > 0 and order_info["state"] == "done":
                    executed_price = current_price
                    await self.trading_history.add_trade(ticker, "buy", executed_price, executed_amount, reason)
                    self.stop_loss_manager.add_position(ticker, executed_price, executed_amount)
                    if ticker not in self.purchased_coins["coins"]:
                        self.purchased_coins["coins"].append(ticker)
                        FileManager.save_json(COINS_FILE, self.purchased_coins)
                    logger.info(f"{ticker} 매수 완료: 가격={executed_price:,.2f}원, 수량={executed_amount:.8f}개")
                    message = (
                        f"🟢 매수 실행: {ticker}\n"
                        f"가격: {executed_price:,.2f}원\n"
                        f"수량: {executed_amount:.8f}개\n"
                        f"총 금액: {executed_amount * executed_price:,.0f}원\n"
                        f"사유: {reason}"
                    )
                    await async_send_telegram_message(message)
                    self.sync_with_upbit_account()
                    return True
                else:
                    logger.error(f"{ticker} 매수 미체결 또는 취소: 상태={order_info['state'] if order_info else '정보 없음'}")
                    if order_info and order_info["state"] == "wait":
                        upbit.cancel_order(order_uuid)
                        logger.info(f"{ticker} 미체결 주문 취소: UUID={order_uuid}")
                        message = (
                            f"⚠️ 매수 미체결 및 취소: {ticker}\n"
                            f"주문 UUID: {order_uuid}\n"
                            f"상태: {order_info['state']}"
                        )
                        await async_send_telegram_message(message)
                    return False
            else:
                logger.error(f"{ticker} 매수 주문 실패: {buy_result or '응답 없음'}")
                return False
        except Exception as e:
            logger.error(f"{ticker} 매수 실행 실패: {e}", exc_info=True)
            return False

    async def log_portfolio_status(self):
        try:
            krw_balance = upbit.get_balance("KRW") or 0
            total_value = float(krw_balance)
            total_invested = 0
            portfolio_message = (
                f"📊 포트폴리오 상태 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
                f"💰 KRW 잔고: {krw_balance:,.2f}원\n"
                f"📋 보유 코인:\n"
            )
            if not self.purchased_coins["coins"]:
                portfolio_message += "없음\n"
            else:
                for ticker in self.purchased_coins["coins"]:
                    balance = upbit.get_balance(ticker.replace("KRW-", ""))
                    if balance and balance > 0:
                        current_price = pyupbit.get_current_price(ticker)
                        if current_price:
                            value = balance * current_price
                            total_value += value
                            position = self.stop_loss_manager.positions.get(ticker, {})
                            avg_price = position.get('entry_price', current_price)
                            invested = balance * avg_price
                            total_invested += invested
                            roi = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
                            portfolio_message += (
                                f"- {ticker}:\n"
                                f"  수량: {balance:.8f}개\n"
                                f"  현재가: {current_price:,.2f}원\n"
                                f"  평균 매수가: {avg_price:,.2f}원\n"
                                f"  가치: {value:,.2f}원\n"
                                f"  수익률: {roi:.2f}%\n"
                            )
                        else:
                            logger.error(f"{ticker} 현재 가격 가져오기 실패")
                            portfolio_message += f"- {ticker}: 가격 정보 없음\n"
                    else:
                        logger.warning(f"{ticker} 잔고 없음 또는 0")
                        portfolio_message += f"- {ticker}: 잔고 없음\n"
            total_roi = ((total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0
            portfolio_message += (
                f"📈 총 투자 금액: {total_invested:,.2f}원\n"
                f"💵 총 자산 가치: {total_value:,.2f}원\n"
                f"📊 총 수익률: {total_roi:.2f}%"
            )
            await async_send_telegram_message(portfolio_message)
            logger.info(f"포트폴리오 상태: KRW 잔고={krw_balance:,.2f}원, 총 투자={total_invested:,.2f}원, 총 가치={total_value:,.2f}원, 수익률={total_roi:.2f}%")
        except Exception as e:
            logger.error(f"포트폴리오 상태 기록 오류: {e}", exc_info=True)
            current_time = time.time()
            if current_time - self.last_portfolio_error_time > 14400:
                await async_send_telegram_message(f"⚠️ 포트폴리오 상태 조회 실패: {str(e)[:100]}...")
                self.last_portfolio_error_time = current_time

    async def run_trading_cycle(self):
        try:
            logger.info("트레이딩 사이클 시작...")
            await asyncio.gather(
                self.check_pending_orders(),
                self.stop_loss_manager.check_limit_orders(),
                self.check_positions(),
                return_exceptions=True
            )
            await self.find_trading_opportunities()
            await self.execute_trading_strategy()
            await self.log_portfolio_status()
            logger.info("트레이딩 사이클 완료")
        except Exception as e:
            logger.error(f"트레이딩 사이클 실패: {e}", exc_info=True)
            await async_send_telegram_message(f"⚠️ 트레이딩 사이클 오류: {str(e)[:100]}...")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @cache_result(expiry_seconds=CACHE_EXPIRY_SECONDS['default'])
    def analyze_coin(self, ticker):
        if not rate_limiter.can_make_api_call():
            time.sleep(0.1)
            return self.analyze_coin(ticker)
        try:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
            if df is None or df.empty:
                logger.error(f"{ticker} OHLCV 데이터 가져오기 실패")
                return None
            if len(df) < 14:
                logger.warning(f"{ticker} 데이터 부족: {len(df)}일, 최소 14일 필요")
                return None
            df = self.market_analyzer.add_technical_indicators(df)
            logger.info(f"{ticker} 코인 분석 완료: {len(df)}일 데이터")
            return df
        except Exception as e:
            logger.error(f"{ticker} 코인 분석 실패: {e}", exc_info=True)
            raise

    # TradingBot 클래스 외부로 이동
    @staticmethod
    async def async_run_bot(cycle_interval=14400):
        """
        트레이딩 봇의 메인 실행 루프. 주기적으로 트레이딩 사이클을 실행하고 포트폴리오 상태를 로깅.
        
        Args:
            cycle_interval (int): 트레이딩 사이클 간 대기 시간(초). 기본값은 3600초(1시간).
        """
        bot = TradingBot()
        try:
            logger.info("초기 포트폴리오 상태 조회 중...")
            await bot.log_portfolio_status()
            while True:
                start_time = time.time()
                await bot.run_trading_cycle()
                elapsed = time.time() - start_time
                remaining = max(0, cycle_interval - elapsed)
                logger.info(f"다음 사이클까지 {remaining:.0f}초 대기")
                if remaining == 0:
                    logger.warning("트레이딩 사이클 실행 시간이 1시간을 초과했습니다. 즉시 다음 사이클 시작.")
                await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            logger.info("프로그램 종료 요청 수신, 리소스 정리 중...")
            raise
        except Exception as e:
            logger.error(f"메인 루프 오류: {str(e)}", exc_info=True)
            await async_send_telegram_message(f"⚠️ 메인 루프 오류: {str(e)[:100]}...")
            logger.info("60초 후 재시도...")
            await asyncio.sleep(60)
            raise

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL", 14400))
    
    logger.info("트레이딩 봇 시작...")
    try:
        asyncio.run(TradingBot.async_run_bot(cycle_interval=CYCLE_INTERVAL))
    except KeyboardInterrupt:
        logger.info("사용자에 의한 종료 요청. 프로그램 종료 중...")
        loop = asyncio.get_event_loop()
        tasks = [task for task in asyncio.all_tasks(loop) if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logger.info("프로그램 정상 종료")
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {str(e)}", exc_info=True)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(async_send_telegram_message(f"🚨 프로그램 비정상 종료: {str(e)[:100]}..."))