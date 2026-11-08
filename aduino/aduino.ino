#include <Servo.h>
#include "EMGFilters.h"

/*
 * ======================================================================================
 * 🦾 5개 서보모터 동시 구동 + EMG 5채널 필터링 송신 + 실시간 제스처 피드백 스케치
 * ======================================================================================
 * - 모터 구동: 표준 <Servo.h> 라이브러리 사용
 * - 동작 정의:
 *     1) 진짜 주먹 (Fist, '1' 수신): L LED(13번) 켜짐 + 5개 서보 한 번에 주먹 쥐기 (ANGLE_FIST = 0도)
 *     2) 아닌 경우 (휴식/잡음, '0' 수신): L LED(13번) 꺼짐 + 5개 서보 '반만 펴기' (ANGLE_HALF_OPEN = 90도)
 * ======================================================================================
 */

// =========================================================================
// ⚙️ [서보모터 핀 설정]
// - 5개 서보 신호선을 1개 핀(9번)에 묶어서 연결하신 경우: {9} (권장: 타이머 인터럽트 부하 최소화)
// - 5개 서보를 서로 다른 핀에 각각 꽂으신 경우: {9, 10, 11, 5, 6} 처럼 해당 핀 번호들을 적으세요.
// =========================================================================
const int SERVO_PINS[] = {9}; // 단일 핀이면 {9}, 개별 핀이면 {9, 10, 11, 5, 6}
const int NUM_SERVOS = sizeof(SERVO_PINS) / sizeof(SERVO_PINS[0]);
Servo servos[NUM_SERVOS];

// =========================================================================
// 📐 [서보모터 동작 각도 설정 (0 ~ 180도)]
// - ANGLE_FIST: 주먹 쥐었을 때 각도 (0도: 와이어 당김 -> 주먹 쥠)
// - ANGLE_HALF_OPEN: 손 폈을 때 각도 (150도: 와이어 완전히 풀림 -> 손가락 펴짐)
// ※ 서보 설치 방향에 따라 반대로 움직인다면 두 값(0과 150)을 맞바꿔주세요.
// =========================================================================
const int ANGLE_FIST = 0;         // 주먹 쥐기 각도 (L LED 켜질 때 -> 접힘/주먹)
const int ANGLE_HALF_OPEN = 150;  // 손 펴기 각도 (L LED 꺼질 때 -> 와이어가 충분히 풀려 손이 펴짐!)

// 핀 설정
const int LED_PIN = LED_BUILTIN; // 13번 내장 LED (아두이노 보드의 'L' 표시등)
const uint8_t pins[] = {A0, A1, A2, A3, A4};
const int numPins = 5;
EMGFilters filters[5];

unsigned long lastMicros = 0;
const unsigned long sampleInterval = 5000; // 200Hz 주기 (5000us)

// 현재 서보 상태 캐시 (-1: 미설정, 0: 반만 펴짐, 1: 주먹 쥠)
// 🚀 상태가 실제로 바뀔 때만 서보를 구동하여 불필요한 인터럽트 및 신호 노이즈 차단!
int currentGripState = -1;

void setGrip(bool isFist) {
  int targetState = isFist ? 1 : 0;
  if (currentGripState == targetState) {
    return; // 상태 변화가 없으면 중복 실행 방지
  }
  currentGripState = targetState;

  if (isFist) {
    // 🚀 진짜 주먹 (Fist) 감지! -> L LED 켜짐 & 서보 접힘(주먹)
    digitalWrite(LED_PIN, HIGH);
    for (int i = 0; i < NUM_SERVOS; i++) {
      servos[i].write(ANGLE_FIST);
    }
  } else {
    // 💤 아닌 경우 (휴식 또는 잡음) -> L LED 꺼짐 & 서보 반만 펴기
    digitalWrite(LED_PIN, LOW);
    for (int i = 0; i < NUM_SERVOS; i++) {
      servos[i].write(ANGLE_HALF_OPEN);
    }
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // 서보모터 연결 및 초기 '반만 펴기' 상태 설정
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(ANGLE_HALF_OPEN);
  }
  currentGripState = 0;

  // EMG 필터 초기화 (기존 원본과 동일)
  for (int i = 0; i < numPins; i++) {
    filters[i].init(SAMPLE_FREQ_500HZ, NOTCH_FREQ_60HZ, true, true, true);
  }
}

void loop() {
  // 1. [PC로부터 AI 추론 결과 실시간 수신] (논블로킹 처리)
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '1') {
      setGrip(true);  // L LED ON + 주먹 쥐기 (0도 접힘)
    } else if (c == '0') {
      setGrip(false); // L LED OFF + 반만 펴기 (90도)
    }
  }

  // 2. [EMG 센서 필터링 및 PC 송신] (200Hz 주기 - 기존 원본 코드 100% 동일)
  unsigned long currentMicros = micros();
  if (currentMicros - lastMicros >= sampleInterval) {
    lastMicros = currentMicros;

    int envs[5];
    for (int i = 0; i < numPins; i++) {
      int raw = analogRead(pins[i]);
      raw = analogRead(pins[i]); // 멀티플렉서 안정화
      int filtered = filters[i].update(raw);
      long env = (long)filtered * filtered; 
      long scaled = env / 20; 
      envs[i] = (scaled > 1023) ? 1023 : scaled;
    }

    // Rust가 파싱하기 쉽도록 CSV 형태로 출력 (예: 120,45,12,5,89)
    Serial.print(envs[0]);
    Serial.print(",");
    Serial.print(envs[1]);
    Serial.print(",");
    Serial.print(envs[2]);
    Serial.print(",");
    Serial.print(envs[3]);
    Serial.print(",");
    Serial.println(envs[4]);
  }
}