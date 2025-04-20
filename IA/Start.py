import pika
import json
from IA_predict import predict  # tu lógica IA aquí

# Conexión al servidor RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters('10.128.0.16'))
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