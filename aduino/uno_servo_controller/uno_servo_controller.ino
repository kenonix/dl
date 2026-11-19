#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

/*
 * ======================================================================================
 * 🦾 [보드 2: 아두이노 Uno 전용] 디지털 트리거 입력 기반 PCA9685 16채널 I2C 서보 컨트롤러
 * ======================================================================================
 * - 역할: ESP32로부터 디지털 핀(D2) 신호(HIGH=주먹 / LOW=펴기)만 받아서 PCA9685 I2C로 서보 5개 구동
 * - 필수 라이브러리: 'Adafruit PWM Servo Driver Library'
 *
 * [배선 안내]
 *   1. ESP32 -> 아두이노 Uno (트리거 신호):
 *      - ESP32 [P23 (GPIO 23)] ---> 아두이노 Uno [D2]
 *      - ESP32 [GND]           <--> 아두이노 Uno [GND] (공통 접지 필수)
 *
 *   2. 아두이노 Uno -> PCA9685 (I2C 서보 드라이버):
 *      - 아두이노 Uno [A4]  ---> PCA9685 [SDA]
 *      - 아두이노 Uno [A5]  ---> PCA9685 [SCL]
 *      - 아두이노 Uno [5V]  ---> PCA9685 [VCC]  (Uno는 5V 보드이므로 5V 연결!)
 *      - 아두이노 Uno [GND] ---> PCA9685 [GND] (및 OE 핀도 GND 연결 권장)
 *      - 외부 5V 파워(+/-)  ---> PCA9685 [V+ / GND] (서보 구동용 분리 전원)
 * ======================================================================================
 */

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40, Wire);

// 트리거 입력 핀 (⚠️ ESP32 3.3V 보호를 위해 반드시 INPUT 사용, INPUT_PULLUP 사용 금지!)
const int TRIGGER_IN_PIN = 2;    // 아두이노 Uno 디지털 2번 핀 (D2)
const int LED_PIN = LED_BUILTIN; // 아두이노 Uno 내장 LED (13번)

// PCA9685 채널 설정 (0~4번: 첫 번째 ~ 다섯 번째 서보)
const uint8_t SERVO_CHANNELS[] = {0, 1, 2, 3, 4};
const int NUM_SERVOS = sizeof(SERVO_CHANNELS) / sizeof(SERVO_CHANNELS[0]);

// 서보 PWM 펄스 범위 (50Hz, 12-bit 0~4095)
const int SERVO_PULSE_MIN = 120; // 0도 펄스
const int SERVO_PULSE_MAX = 490; // 180도 펄스

// 손가락별 개별 각도 (0 ~ 180도)
const int ANGLE_FIST[NUM_SERVOS] = {10, 10, 10, 10, 10};
const int ANGLE_OPEN[NUM_SERVOS] = {95, 95, 95, 95, 95};

int currentGripState = -1; // -1: 미설정, 0: 펴기, 1: 주먹
int lastRawPinState = LOW;
unsigned long lastDebounceTime = 0;
const unsigned long DEBOUNCE_MS = 15; // 15ms 노이즈 디바운싱

uint16_t angleToPulse(int angle) {
  int clamped = constrain(angle, 0, 180);
  return (uint16_t)map(clamped, 0, 180, SERVO_PULSE_MIN, SERVO_PULSE_MAX);
}

void moveAllServos(const int angles[NUM_SERVOS]) {
  for (int i = 0; i < NUM_SERVOS; i++) {
    pwm.setPWM(SERVO_CHANNELS[i], 0, angleToPulse(angles[i]));
  }
}

void setGrip(bool isFist) {
  int targetState = isFist ? 1 : 0;
  if (currentGripState == targetState) {
    return;
  }
  currentGripState = targetState;

  if (isFist) {
    digitalWrite(LED_PIN, HIGH);
    moveAllServos(ANGLE_FIST);
    Serial.println("[UNO] 트리거 HIGH(1) -> 주먹 쥐기");
  } else {
    digitalWrite(LED_PIN, LOW);
    moveAllServos(ANGLE_OPEN);
    Serial.println("[UNO] 트리거 LOW(0) -> 손 펴기");
  }
}

void setup() {
  Serial.begin(9600);

  // ⚠️ 중요: ESP32(3.3V) 출력 핀에 Uno의 5V가 역류하지 않도록 순수 INPUT 모드 사용
  pinMode(TRIGGER_IN_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Wire.begin(); // Uno 기본 I2C: A4(SDA), A5(SCL)
  Wire.beginTransmission(0x40);
  byte err = Wire.endTransmission();
  if (err == 0) {
    Serial.println("[UNO] PCA9685 (0x40) I2C 연결 성공!");
  } else {
    Serial.print("[UNO] PCA9685 (0x40) 응답 없음! A4(SDA)/A5(SCL)/5V/GND 확인, err=");
    Serial.println(err);
  }

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);
  delay(10);

  // 초기 상태: 손 펴기
  moveAllServos(ANGLE_OPEN);
  currentGripState = 0;
  Serial.println("[UNO] 대기 중... (D2 핀 HIGH=주먹, LOW=펴기)");
}

void loop() {
  // 1. [ESP32 -> Uno D2 핀 실시간 감시 (15ms 디바운싱)]
  int rawState = digitalRead(TRIGGER_IN_PIN);
  if (rawState != lastRawPinState) {
    lastDebounceTime = millis();
    lastRawPinState = rawState;
  }

  if ((millis() - lastDebounceTime) >= DEBOUNCE_MS) {
    setGrip(rawState == HIGH);
  }

  // 2. [선택 사항] Uno 시리얼 모니터 단독 테스트 ('1': 주먹, '0': 펴기)
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '1') {
      setGrip(true);
    } else if (c == '0') {
      setGrip(false);
    }
  }
}
