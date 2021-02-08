import csv
import random

a = []
b = []
c = []
d = []
csv_file = csv.reader(open('new.csv'))
#csv_file = csv.reader(open('attachment2_2.csv'))
for content in csv_file:
    content=list(map(float,content))
    if len(content)!=0:
        if content[-1] <= 1.0 and content[-1] >= 0.75:
            a.append(content[-1] + 123)
        elif content[-1] < 0.75 and content[-1] >= 0.50:
            b.append(content[-1] + 123)
        elif content[-1] < 0.50 and content[-1] >= 0.25:
            c.append(content[-1] + 123)
        else :
            d.append(content[-1] + 123)
m = len(a) + len(b) + len(c)
print(len(a))
assignment_sum = 10000
num = assignment_sum/100
num_a = num/4
num_a = int(num_a)
print(num_a)
Q = assignment_sum - num_a * 100
ratio = 0.50
num_bc = m * ratio - num_a
num_bc = int(num_bc)
num_b = int(num_bc/2)
num_c = int(num_b/2)
assignment_b = 80
assignment_c = 55
year = 1
interest_rate = {'A': 0.04,'B': 0.0425,'C': 0.0465}
income = year * interest_rate['A'] * num_a * 100 + year * interest_rate['B'] * num_b * assignment_b + year * interest_rate['C'] * num_c * assignment_c
unassigned = assignment_sum - num_a * 100 - num_b * assignment_b - num_c * assignment_c
print(income)
print(unassigned)
random_a = random.sample(range(0 + 123,len(a) + 123),num_a)
random_b = random.sample(range(0 + 123,len(b) + 123),num_b)
random_c = random.sample(range(0 + 123,len(c) + 123),num_c)
# print(random_a)
# print(random_b)
# print(random_c)
print("random_a:")
for i in range(len(random_a)):
    print(random_a[i], end=' ')
    if (i+1) % 10 == 0:
        print(' ')
print(' ')
print("random_b:")
for i in range(len(random_b)):
    print(random_b[i], end=' ')
    if (i+1) % 10 == 0:
        print(' ')
print(' ')
print("random_c:")
for i in range(len(random_c)):
    print(random_c[i], end=' ')
    if (i+1) % 10 == 0:
        print(' ')