# LSTM-AutoTrade

업비트 API로 시세를 받아 LSTM으로 다음 날 종가를 예측하고, 주문과 손절·익절을 자동으로 처리한 뒤 텔레그램으로 결과를 알리는 암호화폐 자동매매 봇입니다.

> 공개용 기본 구조입니다. 실제로 사용한 종목 선정 기준과 매수 판단 로직은 포함하지 않았으며,
> `scan_for_opportunities`와 `execute_trading_strategy`에는 동작 예시만 들어 있습니다.

## 연구

- 논문: 빅데이터 기반 LSTM 예측 모델을 활용한 암호화폐 자동매매 시스템 연구 (2025, 1저자)
- 2025 한국스마트미디어학회 춘계학술대회 우수논문상
- 2인 프로젝트 (데이터 수집, 매매 로직, 기술적 지표 선정, 텔레그램 연동 담당)

## 처리 흐름

```
시세 수집 → 후보 종목 선정 → LSTM 종가 예측 → 매수 판단 → 주문 → 손절·익절 감시 → 텔레그램 알림
```

정해진 주기(기본 4시간)마다 `run_trading_cycle`이 위 과정을 한 번씩 실행합니다.

## 구성

| 클래스 | 역할 |
|---|---|
| `RateLimiter` | 업비트 API 호출과 주문 횟수를 초 단위로 세고, 한도를 넘으면 기다렸다가 호출 |
| `cache_result` | 같은 요청 결과를 파일에 잠시 저장해 반복 호출을 줄임, 실패 시 재시도 |
| `FileManager` | 보유 종목·거래 내역을 JSON으로 저장 (파일 잠금으로 동시 쓰기 방지) |
| `TradingHistory` | 거래 기록, 손절한 종목의 재매수 금지 목록 관리 |
| `StopLossManager` | 매수가 대비 손절(-5%)·익절(+10%) 주문 관리 |
| `MarketAnalyzer` | LSTM 모델 학습·예측, 기술적 지표 계산 |
| `TradingBot` | 계좌 동기화, 매수·매도 실행, 체결 확인, 보유 현황 알림 |

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # 업비트·텔레그램 키 입력
python autotrading.py
```

실제 계좌로 주문이 나갑니다. 소액으로 먼저 확인하세요.

## 기술

Python, TensorFlow(Keras), PyUpbit, ta, python-telegram-bot, pandas, scikit-learn
