def MapOutput(output): # save Max value in output and check each value for output if equals max then replace it with 1 else 0
    max1 = max(output)
    for i in output:
        if max1 == i:
            i = 1
        else:
            i = 0
    return output

def checkOutput(output, target):
    # [1, 0, 0] == [1, 0, 0] => True other wise is False
    if target.index(max(target)) == output.index(max(output)):
        return True
    else:
        return False

def MapTarget(target): # convert class label "001", "010", "100" to be like [0, 0, 1], [0, 1, 0], [1, 0, 0]
    x = list(target)
    target = list(map(int, x))
    return target

def MapRow(row): # Convert inputs to be like [f1, f2, f3, f4, f5]
    row = list(row)
    return row

def printAcc(acc, data):
    return round((acc / len(data)) * 100, 2) # print Accuracy

def createMatrix(size):
        # This Function Creates the matrix with "-" in each cell
        matrix = [] # make the matrix empty
        # the nested for loops creates the matrix and add "-" in each cell
        for i in range(size ):
            matrix2 = []
            for j in range(size):
                matrix2.append("-")
            matrix.append(matrix2)
        return matrix

def printMatrix(matrix):  # Function to print Confusion Matrix 
    for z in range(len(matrix)):
        print(matrix[z])
    print()