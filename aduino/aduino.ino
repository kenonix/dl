#include "EMGFilters.h"

/*
 * ======================================================================================
 * 🧠 [보드 1: ESP32 전용] 초청정 5채널 EMG 계측 + PC 통신 + 아두이노 트리거 출력
 * ======================================================================================
 * - PCA9685 / I2C / 서보 전원을 ESP32에서 물리적으로 100% 분리하여 EMG 노이즈 원천 차단!
 * - 동작 방식:
 *     1) 5채널 EMG 센서(SVP, SVN, 34, 35, 32)를 200Hz로 필터링하여 PC(Rust)로 전송
 *     2) PC(Rust)로부터 '1'(주먹) 수신 -> 파란 LED(GPIO 2) ON + 트리거 핀(GPIO 23) HIGH(3.3V)
 *     3) PC(Rust)로부터 '0'(휴식/잡음) 수신 -> 파란 LED(GPIO 2) OFF + 트리거 핀(GPIO 23) LOW(0V)
 * - 아두이노 Uno 연결:
 *     * ESP32 [P23 (GPIO 23)] ---> 아두이노 Uno [D2]
 *     * ESP32 [GND]           <--> 아두이노 Uno [GND]
 * ======================================================================================
 */

#ifndef LED_BUILTIN
  #define LED_BUILTIN 2 // ESP32 내장 파란색 LED (GPIO 2)
#endif
const int LED_PIN = LED_BUILTIN;

// 아두이노 Uno(서보 전담 보드)로 파지 신호(1/0)를 보내는 디지털 출력 핀
#ifdef ESP32
  const int TRIGGER_OUT_PIN = 23; // ESP-32S 보드 오른쪽 상단 P23 / IO23 / D23
  // 순수 ADC1 5채널 (보드 표기: SVP(36), SVN(39), P34(34), P35(35), P32(32))
  const uint8_t pins[] = {36, 39, 34, 35, 32};
#else
  const int TRIGGER_OUT_PIN = 9;
  const uint8_t pins[] = {A0, A1, A2, A3, A4};
#endif

const int numPins = 5;
EMGFilters filters[5];

unsigned long lastMicros = 0;
const unsigned long sampleInterval = 5000; // 200Hz 주기 (5000us)

// 현재 출력 상태 캐시 (-1: 미설정, 0: 펴기, 1: 주먹)
int currentGripState = -1;

void setGripTrigger(bool isFist) {
  int targetState = isFist ? 1 : 0;
  if (currentGripState == targetState) {
    return;
  }
  currentGripState = targetState;

  if (isFist) {
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(TRIGGER_OUT_PIN, HIGH); // 아두이노 Uno D2 핀으로 '1' (3.3V HIGH) 전달
  } else {
    digitalWrite(LED_PIN, LOW);
    digitalWrite(TRIGGER_OUT_PIN, LOW);  // 아두이노 Uno D2 핀으로 '0' (0V LOW) 전달
  }
}

void setup() {
  Serial.begin(9600);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pinMode(TRIGGER_OUT_PIN, OUTPUT);
  digitalWrite(TRIGGER_OUT_PIN, LOW);
  currentGripState = 0;

#ifdef ESP32
  // ESP32 10비트(0~1023) 및 11dB 감쇠(0~3.1V) 설정 + 5채널 ADC1 명시 초기화
  analogReadResolution(10);
  analogSetAttenuation(ADC_11db);
  for (int i = 0; i < numPins; i++) {
    pinMode(pins[i], INPUT);
    analogSetPinAttenuation(pins[i], ADC_11db);
  }
#endif

  // EMG 필터 초기화 (60Hz 노치 + 20~150Hz 밴드패스)
  for (int i = 0; i < numPins; i++) {
    filters[i].init(SAMPLE_FREQ_500HZ, NOTCH_FREQ_60HZ, true, true, true);
  }
}

void loop() {
  // 1. [PC로부터 AI 추론 결과 실시간 수신] (논블로킹)
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '1') {
      setGripTrigger(true);  // LED ON + 트리거 핀 HIGH -> 아두이노가 주먹 쥠
    } else if (c == '0') {
      setGripTrigger(false); // LED OFF + 트리거 핀 LOW -> 아두이노가 손 폄
    }
  }

  // 2. [EMG 센서 필터링 및 PC 송신] (200Hz 주기)
  unsigned long currentMicros = micros();
  if (currentMicros - lastMicros >= sampleInterval) {
    lastMicros = currentMicros;

    int envs[5];
    for (int i = 0; i < numPins; i++) {
#ifdef ESP32
      // 8x 오버샘플링으로 ESP32 고주파 ADC 스위칭 잡음 제거
      long mvSum = 0;
      for (int s = 0; s < 8; s++) {
        mvSum += analogReadMilliVolts(pins[i]);
      }
      int mv = (int)(mvSum / 8);
      int raw = (int)((long)mv * 1023L / 3300L);
      int filtered = filters[i].update(raw);
      long env = (long)filtered * filtered;
      // 3.3V -> 5V Uno 기준 스케일 정규화 (필요 시 숫자 조정 가능)
      long scaled = env / 46;
#else
      int raw = analogRead(pins[i]);
      raw = analogRead(pins[i]);
      int filtered = filters[i].update(raw);
      long env = (long)filtered * filtered;
      long scaled = env / 20;
#endif
      envs[i] = (scaled > 1023) ? 1023 : scaled;
    }

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
