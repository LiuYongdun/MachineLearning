

mylist=['male','china','20']

for each in mylist:

    feature=None
    try:
        feature=float(each)
    except:
        feature=each

    print(feature)
