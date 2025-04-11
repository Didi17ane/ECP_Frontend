#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h> 
//#include <HTTPClient.h>

#include <DHT.h>
#define ledpin D4 // Broche D4 pour la led
const char* ssid = "Redmi 12C"; // WE.CODE
const char* password = "Charbel09"; //C0d1ng@2024
const int dhtPin = D2;          // Broche D2 pour le DHT11

// Configuration du capteur DHT
DHT dht(dhtPin, DHT11);

void setup() {
  Serial.begin(115200);
  Serial.println("Demarrage....");
  pinMode(ledpin, OUTPUT);
  
  // Initialisation du DHT11
  dht.begin();
  
  //Connexion au Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connexion à ");
  Serial.print(ssid);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  
  Serial.println("\nConnecté au Wi-Fi");
  //Serial.println(WiFi.localIP());
}

void loop() {
  // Lire les valeurs du DHT11
  float temperature = dht.readTemperature();
  float humidite = dht.readHumidity();
  
  // Visualiser les données réceuillies dans le moniteur série

  Serial.println("ENVOI DES INFORMATIONS : ");
  Serial.print("Température : ");
  Serial.print(temperature);
  Serial.println("°C");

  Serial.print("Humidité : ");
  Serial.print(humidite);
  Serial.println("%");

  // Envoi des données au serveur

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    WiFiClient client;
    http.begin(client, "http://192.168.111.73:8000/api/dht/"); // URL de l'API
    http.addHeader("Content-Type", "application/json");
    
    String params = "{\"temperature\": " + String(temperature) +
                    ", \"humidite\": " + String(humidite)+"}";
    
    int httpResponseCode = http.POST(params);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println(httpResponseCode);
      if (httpResponseCode == 201) {
        // Clignotement LED
        digitalWrite(ledpin, HIGH);
        delay(500);
        digitalWrite(ledpin, LOW);
        delay(500);
      }
      //Serial.println(response);
    } else {
      Serial.print("Erreur lors de la requête : ");
      Serial.println(httpResponseCode);
    }
    
    http.end(); // Libération des ressources
  }

  delay(1000); // Attendre 1 seconde avant de lire à nouveau
}
