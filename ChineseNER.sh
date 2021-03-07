#! /bin/bash
QUIT_COMMAND=n
# ֱ���û�������ַ�����quitʱ��untilѭ���Ż��˳�
until [ "$USER_INPUT" = "$QUIT_COMMAND" ]
do
    read -p ">> Please input a number to select a method(1.crfsuite 2.Lattice LSTM 3.Bert BiLSTM CRF 4.BERT-IDCNN-LSTM-CRF):" NUM1
    # ����ѡ��Ϊ��
    if [ ! -n "$NUM1" ]
    then
        echo "[ERROR] you have not input a word!"
    # ����ѡ��Ϊ1����Ӧcrfsuiteģ��
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
    # ����ѡ��Ϊ2����ӦLattice LSTMģ��
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
    # ����ѡ��Ϊ3����ӦBert LSTM CRFģ��
    elif [ $NUM1 -eq 3 ]
    then
        read -p ">> Please input a sentence:" S2
        if [ ! -n "$S2" ]
        then
            echo "[ERROR] you have not input a word!"
        else
            python predict.py $S2
        fi
    #����ѡ��Ϊ4����ӦBert Idcnn LSTM CRFģ��
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
    # �ж��Ƿ��������ʶ��
    read -p ">> Do you want to continue?(y/n):" USER_INPUT
done