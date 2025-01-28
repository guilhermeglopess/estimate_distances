28 janeiro 2025

bem vindos ao meu readme

informacoes:
modelo DP - modelo detecao de pessoas
modelo DA - modelo depth anything
modelo ED - modelo estimacao de distancias (este nem é bem um modelo isto so usa os outros dois e faz umas contas manhosas para ajustar)
o modelo ED resulta na utilizacao conjunta dos modelos DP e DA
tudo que fiz neste repositorio está na pasta gui, tudo o resto já vinha do modelo DA
dentro da pasta gui ha o ficheiro estimate.py, é esse que devem correr
dentro da pasta gui ha a pasta modelo_pessoas que tem o best.pt do modelo DP
dentro da pasta gui ha a pasta not_used para guarda coisas que ja nao sao precisas ou que poderao vir a ser mas nao estao prontas


procedimento do modelo ED (por agora):
em cada frame, estimar a distancia de cada ponto do frame a camera com uma inferencia do modelo DA (nao é direto, mas podem ver as contas feitas nos comentarios da funcao)
a inferencia do modelo DA devolve uma matriz DEPTH, x por y onde cada entrada tem o valor da distancia do ponto x,y à camera (x e y - largura e altura do frame)
em cada frame, fazer uma inferencia do modelo DP e guardar detecoes de pessoas de cada frame numa lista 
para cada detecao feita, calcular o seu centro: (x1, y1)
colocar uma bounding box a volta da pessoa com o label DEPTH[x1, y1] (entrada x1,y1 da matriz DEPTH) 

------------------------------------


