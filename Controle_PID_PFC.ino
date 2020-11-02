#include <AutoPID.h>

const byte MOTOR_A = 3;  // Motor A Interrupt Pin - INT 1 - Motor DIreita
const byte MOTOR_B = 2;  // Motor B Interrupt Pin - INT 0 - Motor Esquerda

// Contadores de pulsos de mudança nos encoders
double counter_D = 0;
double counter_E = 0;

// Motor A

int enA = 10;
int in1 = 13;
int in2 = 12;

// Motor B

int enB = 9;
int in3 = 8;
int in4 = 7;

// Interrupt Service Routines

// Rotina de tratamento de Interrupção do Motor A
void ISR_countA()  
{
  counter_D++;  // Incrementa o valor do contador A
} 

// Rotina de tratamento de Interrupção do Motor B
void ISR_countB()  
{
  counter_E++;  // Incrementa o valor do contador B
}

#define outputmin_E 40
#define outputmax_E 170
#define outputmin_D 40
#define outputmax_D 200
#define KP 20
#define KI 2.4
#define KD 0.6

#define sampletime 400
double output_E, output_D;
double target_t = -190;

AutoPID myPID_E(&counter_E, &target_t, &output_E, outputmin_E, outputmax_E, KP, KI, KD);

AutoPID myPID_D(&counter_D, &target_t, &output_D, outputmin_D, outputmax_D, KP, KI, KD);

char buf;

void setup() {
  // put your setup code here, to run once:
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  attachInterrupt(digitalPinToInterrupt (MOTOR_A), ISR_countA, RISING);  // Incrementa counter A quando o pino do sensor de velocidade entra em nível alto
  attachInterrupt(digitalPinToInterrupt (MOTOR_B), ISR_countB, RISING);  // Incrementa counter A quando o pino do sensor de velocidade entra em nível alto
  Serial.begin(115200);
  myPID_E.setTimeStep(400);
  myPID_D.setTimeStep(400);
}
void loop() {
  // put your main code here, to run repeatedly:
    buf = Serial.read();
    double target = 140;
    if (buf == 'E'){ //Esquerda
      analogWrite(enA, 100);
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      analogWrite(enB, 100);
      digitalWrite(in3, LOW);
      digitalWrite(in4, HIGH);
      target_t = 0;
      myPID_E.reset();
      myPID_D.reset();
      counter_D = 0;
      counter_E = 0;
      }
    else if (buf == 'D'){ //Direita
      analogWrite(enA, 100);
      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      analogWrite(enB, 100);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      target_t = 0;
      myPID_E.reset();
      myPID_D.reset();
      counter_D = 0;
      counter_E = 0;
      }
   else if (buf == 'F'){ //Frente
      target_t += target;
      myPID_E.run();
      myPID_D.run();
      analogWrite(enA, output_D);
      analogWrite(enB, output_E);
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
      }
    else if (buf == 'T'){ //Tras
      target_t += target;
      myPID_E.run();
      myPID_D.run();
      analogWrite(enA, output_D);
      analogWrite(enB, output_E);
      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      digitalWrite(in3, LOW);
      digitalWrite(in4, HIGH);
      }
    else if (buf == 'P'){ //Parado
      analogWrite(enA, 0);
      analogWrite(enB, 0);
      target_t = 0;
      myPID_E.reset();
      myPID_D.reset();
      counter_D = 0;
      counter_E = 0;
      }
}
