from math import pi

fp = open("data/data_position.txt", "r")
fv = open("data/data_vitesse.txt", "r")
fe = open("data/data_effort.txt", "r")
fa = open("data/data_all.txt", "w")

fa.write("q1 q2 q3 q4 q5 q6 dq1 dq2 dq3 dq4 dq5 dq6 t1 t2 t3 t4 t5 t6\n")
lp = fp.readline()
lv = fv.readline()
le = fe.readline()
while(len(lp)!=0):
    la = [float(i) for i in lp.split()] + [float(i)*pi/180 for i in lv.split()] + [float(i) for i in le.split()] 
    fa.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        la[0],la[1],la[2],la[3],la[4],la[5],la[6],la[7],la[8],la[9],la[10],la[11],la[12],la[13],la[14],la[15],la[16],la[17],
    ))
    lp = fp.readline()
    lv = fv.readline()
    le = fe.readline()   
fp.close()
fv.close()
fe.close()
fa.close()