import pika, ssl
import json
from IA_predict import predict  # tu lógica IA aquí
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# Clave de cifrado (debe ser de 16, 24 o 32 bytes)
KEY = b'abcdefghijklmnop'
IV = b'0000000000000000'  # Vector de inicialización

def encrypt_json(data):
    cipher = AES.new(KEY, AES.MODE_CBC, IV)
    json_str = json.dumps(data)  # Convierte JSON a string
    encrypted_bytes = cipher.encrypt(pad(json_str.encode(), AES.block_size))
    return base64.b64encode(encrypted_bytes).decode()

def decrypt_json(encrypted_json):
    encrypted_bytes = base64.b64decode(encrypted_json)
    cipher = AES.new(KEY, AES.MODE_CBC, IV)
    decrypted_bytes = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)
    return json.loads(decrypted_bytes.decode())

# Conexión al servidor RabbitMQ
rabbit_host = '10.128.0.16'
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