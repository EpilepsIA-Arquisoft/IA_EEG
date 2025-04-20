import pika
import json
from IA_predict import predict  # tu lógica IA aquí

# Conexión al servidor RabbitMQ
rabbit_host = '10.128.0.16'
rabbit_user = 'isis2503'
rabbit_password = '1234'
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=rabbit_host,
                              credentials=pika.PlainCredentials(rabbit_user, rabbit_password)))
channel = connection.channel()

# Asegurarse de que la cola exista
channel.queue_declare(queue='ia_requests')
channel.queue_declare(queue='ia_responses')

def callback(ch, method, properties, body):
    entrada = json.loads(body)
    resultado = predict(entrada)
    
    # Enviar resultado a otra cola
    channel.basic_publish(
        exchange='',
        routing_key='ia_responses',
        body=json.dumps(resultado)
    )

    ch.basic_ack(delivery_tag=method.delivery_tag)

# Escuchar los mensajes entrantes
channel.basic_consume(queue='ia_requests', on_message_callback=callback)
print("Esperando mensajes...")
channel.start_consuming()