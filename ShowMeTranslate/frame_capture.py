import cv2
import os
from classification import image_classification

# Usa a biblioteca Opencv Python para ler o vídeo e
# capturar alguns frames em imagens

def frame_capture(video_name):
    frame_list = []
    # Limpa o diretório de imagens
    image_list = os.listdir('./imgs')
    for img in image_list:
        os.remove('./imgs/'+img)
    # Utiliza o vídeo que estiver dentro do diretório "video"
    video = cv2.VideoCapture('./video/'+str(video_name)+'.mp4')
    counter = 0

    while True:
        check, frame = video.read()

        if not check:
            break
        # Pega um frame a cada 15 frames
        if (counter > 0 and counter % 15 == 0):
            resize = cv2.resize(frame, (480, 640))
            
            # Cria uma imagem do frame atual
            cv2.imwrite('./imgs/'+str(counter)+'.png', frame)

            # Busca as imagens dentro da pasta 'imgs'
            image_list = os.listdir('./imgs')

            # Busca a classificação do frame atual
            frames_result = image_classification('./imgs/'+str(image_list[-1]))

            # O gesto 'parada' ainda não está implementado, então é ignorado
            if frames_result != 'parada':
                frame_list.append(frames_result)

            # Escreve no vídeo qual o gesto daquele frame
            cv2.putText(resize, frames_result,(10,25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            cv2.imshow('Analisando...', resize)
            key = cv2.waitKey(10)
            # A tecla 'q' do teclado encerra o programa
            if key==ord('q'):
                break
        counter += 1

    video.release()
    cv2.destroyAllWindows
    return frame_list