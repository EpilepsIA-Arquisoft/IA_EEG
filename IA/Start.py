import pika
import json
from IA_predict import predict  # tu lógica IA aquí
import base64
import hashlib
import hmac
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# Claves secretas
KEY = b'abcdefghijklmnop'  # AES key (16 bytes)
HMAC_KEY = b'supersecretkey123'  # HMAC key

IV = b'0000000000000000'  # Vector de inicialización

def encrypt_json(data):
    # Convertir JSON a cadena
    json_str = json.dumps(data)

    # Cifrar usando AES-CBC
    cipher = AES.new(KEY, AES.MODE_CBC, IV)
    encrypted_bytes = cipher.encrypt(pad(json_str.encode(), AES.block_size))
    encrypted_base64 = base64.b64encode(encrypted_bytes).decode()

    # Generar HMAC para integridad
    hmac_signature = hmac.new(HMAC_KEY, encrypted_bytes, hashlib.sha256).hexdigest()

    return json.dumps({"data": encrypted_base64, "hmac": hmac_signature})

def decrypt_json(encrypted_json):
    encrypted_json = json.loads(encrypted_json)
    encrypted_base64 = encrypted_json["data"]
    received_hmac = encrypted_json["hmac"]

    # Decodificar base64
    encrypted_bytes = base64.b64decode(encrypted_base64)

    # **Verificar integridad** comparando HMAC
    expected_hmac = hmac.new(HMAC_KEY, encrypted_bytes, hashlib.sha256).hexdigest()
    
    if expected_hmac != received_hmac:
        raise ValueError("Error: La integridad del mensaje ha sido comprometida")

    # Descifrar mensaje
    cipher = AES.new(KEY, AES.MODE_CBC, IV)
    decrypted_bytes = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)
    
    return json.loads(decrypted_bytes.decode())

# Conexión al servidor RabbitMQ
rabbit_host = '10.128.0.20'
rabbit_user = 'isis2503'
rabbit_password = '1234'
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=rabbit_host,
                              credentials=pika.PlainCredentials(rabbit_user, rabbit_password)))
channel = connection.channel()

# Asegurarse de que la cola exista
channel.queue_declare(queue='ia_requests', durable=True, exclusive=False, auto_delete=False)
channel.queue_declare(queue='ia_responses', durable=True, exclusive=False, auto_delete=False)

channel.basic_qos(prefetch_count=1)  # Asegurarse de que solo se procese un mensaje a la vez

def callback(ch, method, properties, body):
    try:
        entrada = decrypt_json(body)
        resultado = predict(entrada)
        
        # Enviar resultado a otra cola
        channel.basic_publish(
            exchange='',
            routing_key='ia_responses',
            body= encrypt_json(resultado),
            properties=pika.BasicProperties(
                delivery_mode=2  # 1 = no persistente, 2 = persistente
            )
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("Esperando mensajes...")
    except Exception as e:
        print("Error procesando el mensaje:", e)
        

# --> Escuchar los mensajes entrantes
channel.basic_consume(
    queue='ia_requests', 
    on_message_callback=callback)

print("Esperando mensajes...")
channel.start_consuming()