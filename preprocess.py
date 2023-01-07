import numpy as np
import pandas as pd

def preprocess(): # Function to Read dataset and count Male and Female in each class to replace Null values with the
    # one which has more occured in that class
    # after that replace Male with 1 and Female with -1
    df = pd.read_csv("penguins.csv")
    AdelieClass = df[(df['species'] == 'Adelie')]
    GentooClass = df[(df['species'] == 'Gentoo')]
    ChinstrapClass = df[(df['species'] == 'Chinstrap')]
    AdelieMaleCtn = AdelieClass['gender'].value_counts()["male"]
    AdelieFemaleCtn = AdelieClass['gender'].value_counts()["female"]
    GentooMaleCtn = GentooClass['gender'].value_counts()["male"]
    GentooFemaleCtn = GentooClass['gender'].value_counts()["female"]
    ChinstrapMaleCtn = ChinstrapClass['gender'].value_counts()["male"]
    ChinstrapFemaleCtn = ChinstrapClass['gender'].value_counts()["female"]
    if AdelieMaleCtn > AdelieFemaleCtn:
        AdelieClass.fillna("male", inplace=True)
    else:
        AdelieClass.fillna("female", inplace=True)
    if GentooMaleCtn > GentooFemaleCtn:
        GentooClass.fillna("male", inplace=True)
    else:
        GentooClass.fillna("female", inplace=True)
    if ChinstrapMaleCtn > ChinstrapFemaleCtn:
        ChinstrapClass.fillna("male", inplace=True)
    else:
        ChinstrapClass.fillna("female", inplace=True)
    AdelieClass.replace("male", 1, inplace=True)
    AdelieClass.replace("female", -1, inplace=True)
    AdelieClass.replace("Adelie", "100", inplace=True) # [1, 0, 0]
    GentooClass.replace("male", 1, inplace=True)
    GentooClass.replace("female", -1, inplace=True)
    GentooClass.replace("Gentoo", "010", inplace=True)
    ChinstrapClass.replace("male", 1, inplace=True)
    ChinstrapClass.replace("female", -1, inplace=True)
    ChinstrapClass.replace("Chinstrap", "001", inplace=True)
    return AdelieClass, GentooClass, ChinstrapClass


def fitdata():
    AClass, GClass, CClass = preprocess()
    # Train Data
    TrainSet = AClass[:30]
    TrainSet = TrainSet.append(GClass[:30], ignore_index=True)
    TrainSet = TrainSet.append(CClass[:30], ignore_index=True)
    TrainLables = TrainSet["species"]
    TrainData = TrainSet.drop(columns=["species"])
    # Test Data
    TestSet = AClass[30:]
    TestSet = TestSet.append(GClass[30:], ignore_index=True)
    TestSet = TestSet.append(CClass[30:], ignore_index=True)
    TestLabels = TestSet["species"]
    TestData = TestSet.drop(columns=["species"])
    return TrainLables, TrainData, TestLabels, TestData