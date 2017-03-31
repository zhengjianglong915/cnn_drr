
def dr_evaluate(test_label, predict_label):
    TP = 0
    FP = 0
    FN = 0
    acc_count = 0
    for test_v,  predict_v in zip(test_label, predict_label):
        if test_v == 1:
            if predict_v == 1:
                TP = TP + 1
            else:
                FN = FN + 1
        if predict_v == 1 and test_v == 0:
            FP = FP + 1 
       
        if test_v ==  predict_v:
            acc_count = acc_count + 1
        
    acc = (acc_count * 1.0) / len(test_label)    
    recall = (TP * 1.0) / (TP + FN + 0.0000001)
    pre = (TP * 1.0) / (TP + FP + 0.0000001)
    f_measure = (2 * recall * pre) / (recall + pre + 0.0000001)
    return f_measure, pre, recall, acc

    
