# dacon_Computer_Vision

어디 한번 10위권을 노려보자고~ 100만원 받으면 개꿀이겠지만 ㅋㅋㅋㅋㅋ 세상에 실력자들은 넘처난다..



1. Efficientnet_b4 학습 f1 score = 76%
2. Efficientnet_b4, good을 제외한 나머지 데이터 90도, -90도, 180도, 반전 증강하여 학습 f1 score = 78%
3. Efficientnet_b4, good을 제외한 나머지 데이터 90도, -90도, 180도, 반전 증강, image_resize (512, 512) -> (700, 700) 학습 f1 score = 81.02%
4. Efficientnet_b4, good을 제외한 나머지 데이터 90도, -90도, 180도, 반전 증강, image_resize (512, 512) -> (700, 700), normalize dataset 추가하여 학습 f1 score = 80%
5. Efficientnet_b4, good을 제외한 나머지 데이터 90도, -90도, 180도, 반전 증강, image_resize (512, 512) -> (700, 700) 학습 f1 score = 81.03%
