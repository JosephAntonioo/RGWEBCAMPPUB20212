import time
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile

pose_path = "pose.zip"
zip_object = zipfile.ZipFile(file=pose_path, mode="r")
zip_object.extractall("./")

imagens_path = "../../../PycharmProjects/ReconhecimentoDeGestos/imagens.zip"
zip_object = zipfile.ZipFile(file=imagens_path, mode="r")
zip_object.extractall("./")

modulos_path = "modulos.zip"
zip_object = zipfile.ZipFile(file=modulos_path, mode="r")
zip_object.extractall("./")
zip_object.close()

sys.path.append('modulos/')
sys.path

import modulos.extrator_POSICAO as posicao
import modulos.extrator_ALTURA as altura
import modulos.extrator_PROXIMIDADE as proximidade
import modulos.alfabeto as alfabeto

arquivo_proto = "pose/hand/pose_deploy.prototxt"
arquivo_pesos = "pose/hand/pose_iter_102000.caffemodel"
numero_pontos = 22
pares_pose = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
              [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
              [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P',
          'Q', 'R', 'S', 'T', 'U', 'V', 'W']

modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)

cor_pontoA, cor_pontoB, cor_linha = (14, 201, 255), (255, 0, 128), (192, 192, 192)
cor_txtponto, cor_txtinicial, cor_txtandamento = (10, 216, 245), (255, 0, 128), (192, 192, 192)

tamanho_fonte, tamanho_linha, tamanho_circulo, espessura = 1, 1, 4, 2
fonte = cv2.FONT_HERSHEY_SIMPLEX

video = "imagens/hand/Libras/libras_4.mp4"
captura = cv2.VideoCapture(0)
ret, frame = captura.read()
print(ret)

imagem_largura = frame.shape[1]
imagem_altura = frame.shape[0]
proporsao = imagem_largura / imagem_altura

print(str(imagem_largura) + " " + str(imagem_altura) + " " + str(proporsao))

entrada_altura = 368
entrada_largura = int(((proporsao * entrada_altura) * 8) // 8)
print(str(entrada_largura) + " " + str(entrada_altura))

resultado = './libras.avi'
gravar_video = cv2.VideoWriter(resultado, cv2.VideoWriter_fourcc(*'XVID'), 10,
                              (frame.shape[1], frame.shape[0]))

modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)

limite = 0.1
while (cv2.waitKey(1) < 0):
    t = time.time()
    conectado, frame = captura.read()
    frame_copia = np.copy(frame)

    tamanho = cv2.resize(frame, (imagem_largura, imagem_altura))
    mapa_suave = cv2.GaussianBlur(tamanho, (3, 3), 0, 0)
    fundo = np.uint8(mapa_suave > limite)

    if not conectado:
        cv2.waitKey()
        break

    entrada_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255,
                                         (entrada_largura, entrada_altura),
                                         (0, 0, 0), swapRB=False, crop=False)

    modelo.setInput(entrada_blob)

    saida = modelo.forward()

    pontos = []

    for i in range(numero_pontos):
        mapa_confianca = saida[0, i, :, :]
        mapa_confianca = cv2.resize(mapa_confianca, (imagem_largura, imagem_altura))

        _, confianca, _, point = cv2.minMaxLoc(mapa_confianca)

        if confianca > limite:
            cv2.circle(frame_copia, (int(point[0]), int(point[1])),
                       tamanho_circulo, cor_pontoA, thickness=espessura,
                       lineType=cv2.FILLED)
            cv2.putText(frame_copia, "{}".format(i), (int(point[0]), int(point[1])),
                        fonte, .8,
                        cor_txtponto, 2, lineType=cv2.LINE_AA)

            pontos.append((int(point[0]), int(point[1])))
        else:
            pontos.append((0, 0))

    for par in pares_pose:
        parteA = par[0]
        parteB = par[1]

        if pontos[parteA] != (0, 0) and pontos[parteB] != (0, 0):
            cv2.line(frame, pontos[parteA], pontos[parteB], cor_linha,
                     tamanho_linha, lineType=cv2.LINE_AA)
            cv2.circle(frame, pontos[parteA], tamanho_circulo, cor_pontoA,
                       thickness=espessura, lineType=cv2.FILLED)
            cv2.circle(frame, pontos[parteB], tamanho_circulo, cor_pontoB,
                       thickness=espessura, lineType=cv2.FILLED)

            cv2.line(fundo, pontos[parteA], pontos[parteB], cor_linha,
                     tamanho_linha, lineType=cv2.LINE_AA)
            cv2.circle(fundo, pontos[parteA], tamanho_circulo, cor_pontoA,
                       thickness=espessura, lineType=cv2.FILLED)
            cv2.circle(fundo, pontos[parteB], tamanho_circulo, cor_pontoB,
                       thickness=espessura, lineType=cv2.FILLED)

    posicao.posicoes = []

    # dedo polegar
    posicao.verificar_posicao_DEDOS(pontos[1:5], 'polegar', altura.verificar_altura_MAO(pontos))

    # dedo indicador
    posicao.verificar_posicao_DEDOS(pontos[5:9], 'indicador', altura.verificar_altura_MAO(pontos))

    # dedo médio
    posicao.verificar_posicao_DEDOS(pontos[9:13], 'medio', altura.verificar_altura_MAO(pontos))

    # dedo anelar
    posicao.verificar_posicao_DEDOS(pontos[13:17], 'anelar', altura.verificar_altura_MAO(pontos))

    # dedo mínimo
    posicao.verificar_posicao_DEDOS(pontos[17:21], 'minimo', altura.verificar_altura_MAO(pontos))

    for i, a in enumerate(alfabeto.letras):
        if proximidade.verificar_proximidade_DEDOS(pontos) == alfabeto.letras[i]:
            cv2.putText(frame, 'Letra: ' + letras[i], (50, 50), fonte, 1,
                        cor_txtinicial, tamanho_fonte,
                        lineType=cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Analisando', (250, 50), fonte, 1,
                        cor_txtandamento, tamanho_fonte,
                        lineType=cv2.LINE_AA)

    cv2.imshow('', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    print("Tempo total = {:.2f}seg".format(time.time() - t))
    gravar_video.write(frame)
gravar_video.release()
