from sign_predict import sign_classification
from classification import image_classification, initialize_tensorflow
from frame_capture import frame_capture

if __name__ == '__main__':
    print('Iniciando...')
    video_name = input('Digite o nome do vídeo: ')
    initialize_tensorflow()
    classification_list = frame_capture(video_name)
    result = sign_classification(classification_list)
    print('Resultado: ' +str(result[0]))