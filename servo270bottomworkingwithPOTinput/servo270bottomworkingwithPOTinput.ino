#include <Servo.h>

Servo myservo;

const int potPin = A0;    // Analog pin connected to the potentiometer
const int servoPin = 9;   // Digital pin connected to the servo signal wire

void setup() {
  myservo.attach(servoPin);  // Attach the servo to pin 9
}

void loop() {
  int potValue = analogRead(potPin);  // Read potentiometer value (0-1023)

  // Map the potentiometer value to the servo pulse width (500us to 2500us)
  int pulseWidth = map(potValue, 0, 1023, 500, 2500);

  myservo.writeMicroseconds(pulseWidth);  // Set servo position
  delay(15);  // Small delay for servo to move to the position
}

