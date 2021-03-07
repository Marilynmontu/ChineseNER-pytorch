#! /bin/bash
QUIT_COMMAND=n
# 直到用户输入的字符串是quit时，until循环才会退出
until [ "$USER_INPUT" = "$QUIT_COMMAND" ]
do
    read -p ">> Please input a number to select a method(1.crfsuite 2.Lattice LSTM 3.Bert BiLSTM CRF 4.BERT-IDCNN-LSTM-CRF):" NUM1
    # 输入选择为空
    if [ ! -n "$NUM1" ]
    then
        echo "[ERROR] you have not input a word!"
    # 输入选择为1，对应crfsuite模型
    elif [ $NUM1 -eq 1 ]
    then
        read -p ">> Please input a sentence:" S2
        if [ ! -n "$S2" ]
        then
            echo "[ERROR] you have not input a word!"
        else 
            isTrain=0
            python crfsuite.py $S2 $isTrain
        fi
    # 输入选择为2，对应Lattice LSTM模型
    elif [ $NUM1 -eq 2 ]
    then
        read -p ">> Please input a sentence:" S2
        if [ ! -n "$S2" ]
        then
            echo "[ERROR] you have not input a word!"
        else
            python Wrapper.py $S2  &> /dev/null
            cat ./data/output.format
        fi
    # 输入选择为3，对应Bert LSTM CRF模型
    elif [ $NUM1 -eq 3 ]
    then
        read -p ">> Please input a sentence:" S2
        if [ ! -n "$S2" ]
        then
            echo "[ERROR] you have not input a word!"
        else
            python predict.py $S2
        fi
    #输入选择为4，对应Bert Idcnn LSTM CRF模型
    elif [ $NUM1 -eq 4 ]
    then
        read -p ">> Please input a sentence:" S2
        if [ ! -n "$S2" ]
        then
            echo "[ERROR] you have not input a word!"
        else
            python Wrapper.py $S2
#            cat ./data/output.format
        fi
    fi
    # 判断是否继续进行识别
    read -p ">> Do you want to continue?(y/n):" USER_INPUT
done