#简单的复利计算

principal= 1000;#初始金额
rate =0.05; #利率
numyears= 5;# 年数
year =1 ;
while year <= numyears:
    principal =principal*(1+rate);
    print(year, principal);
    print("%3d %0.2f" % (year, principal),end='')
    print(format(year,"5d"),format(principal,"0.2f"))
    year += 1


for line in open("foo.txt"):
    print(line)


f = open("out",'w')
#while year <= numyears:
#    principal = principal * (1+rate)
#    print("%3d %0.2f" % (year, principal), file=f)
f.write("a1 a2")
f.writelines([str(1),str(2)])
 #   year += 1
f.close()
print((str(1),str(2)))





