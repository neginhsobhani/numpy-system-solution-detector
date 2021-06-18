import numpy as np


def print_matrix(matrix):
    for row in range(matrix.shape[0]):
        if row == 0:
            print('[[', end=' ')
        else:
            print(' [', end=' ')
        for col in range(matrix.shape[1]):
            print(round(matrix[row][col], 1), end='   \t')
        if row == matrix.shape[0] - 1:
            print(']]')
        else:
            print(']')


def print_equations(row, column, matrix):
    for r in range(0, row):
        print('    row', end='')
        print(r + 1, ': ', end='')
        left_side_is_zero = True
        for col in range(0, column - 1):
            if matrix[r][col] != 0:
                if not left_side_is_zero:
                    print('+ ', end='')
                left_side_is_zero = False
                print('x', end='')
                print(col + 1, end='')
                print('(', end='')
                print(round(matrix[r][col], 2), end=')\t')
        if left_side_is_zero:
            if round(matrix[r][-1]) != 0:
                print('0 ', end='')
                print('=', round(matrix[r][-1], 2))
            else:
                print('0 ', end='')
                print('=', round(matrix[r][-1], 2))
        else:
            print('=', round(matrix[r][-1], 2))
    print('')


def iszero(row, column, matrix):
    counter = 0
    for c in range(column):
        if matrix[row, c] == 0:
            counter += 1
    if counter == column:
        return True
    return False


# exchange zero line - moves zero rows to the end of the matrix
def exchangezerorows(row, column, matrix):  # column is the number of matrix columns
    for r in range(row):
        if iszero(r, column, matrix):
            for n in range(row - 1, r, -1):
                if iszero(n, column, matrix):
                    continue
                print("exchange - zero row")
                res = exchange(row, r, n, matrix)
                return res
    return matrix


# ELEMENTARY OPERATION 1 - row exchange
# exchange any two rows - org is the upper row and des is the lower row  - exchange org and des
def exchange(row, org, des, matrix):
    e = np.identity(row)
    e[[org, des], :] = e[[des, org], :]
    print("elementary matrix - row exchange:")
    print_matrix(e)
    matrix[[org, des]] = matrix[[des, org]]
    print("Current matrix after row operation")
    print_matrix(matrix)
    print("Current equation system :")
    print_equations(row, matrix.shape[1], matrix)
    print("-----------------------------------------")
    return matrix


# ELEMENTARY OPERATION 2
# multiply a row in a number
def scaling(totalrow, row, num, matrix):  # row is multiplied by num
    e = np.identity(totalrow)
    e[row, :] *= num
    print("elementary matrix - scaling:")
    print_matrix(e)
    scaled_row = matrix[row, :] * num
    matrix[row, :] = scaled_row
    print("Current matrix after row operation")
    print_matrix(matrix)
    print("Current equation system :")
    print_equations(totalrow, matrix.shape[1], matrix)
    print("-----------------------------------------")
    return matrix


# ELEMENTARY OPERATION 3
# add multiply of a upper row to a lower one
def replacement(totalrow, org, des, num, matrix):
    e = np.identity(totalrow)
    temp = e[org, :] * num + e[des, :]
    e[des, :] = temp
    print("elementary matrix - replacement:")
    print_matrix(e)
    modified_row = matrix[org, :] * num + matrix[des, :]
    matrix[des, :] = modified_row
    print("Current matrix after row operation")
    print_matrix(matrix)
    print("Current equation system :")
    print_equations(totalrow, matrix.shape[1], matrix)
    print("-----------------------------------------")
    return matrix


def find_pivots(level, row, column, matrix):
    pivots = []
    for r in range(level, row, 1):
        if iszero(r, column, matrix):
            continue
        for c in range(column):
            if matrix[r, c] == 0:
                continue
            pivots.append(c)
            break
    return pivots


def swap_list_elements(i, j, list):
    list[i], list[j] = list[j], list[i]


def swap_row(i, j, matrix):
    matrix[[i, j]] = matrix[[j, i]]


def arrange_rows(level, row, column, matrix):
    pivots = find_pivots(level, row, column, matrix)
    for i in range(len(pivots)):
        for j in range(i, len(pivots), 1):
            if pivots[i] > pivots[j]:
                swap_list_elements(i, j, pivots)
                swap_row(i, j, matrix)
    return pivots


def zero_under_rows(currentrow, row, currentcolumn, matrix):
    for r in range(currentrow + 1, row, 1):
        if matrix[r, currentcolumn] != 0 and matrix[currentcolumn, currentcolumn] != 0:
            zarib = matrix[r, currentcolumn] / matrix[currentcolumn, currentcolumn]
            replacement(row, currentrow, r, -zarib, matrix)


def echelon_form(row, column, matrix):
    for level in range(row):
        exchangezerorows(row, column, matrix)
        pivots = arrange_rows(level, row, column, matrix)
        if len(pivots) != 0:
            zero_under_rows(level, row, pivots[0], matrix)
    print(" >> ECHELON FORM << ")
    print_matrix(matrix)
    print("******************************************")
    return matrix


# the process of finding reduced echelon form
def zero_upper_rows(currentrow, row, currentcolumn, matrix):
    for r in range(0, currentrow, 1):
        if matrix[r, currentcolumn] != 0:
            zarib = matrix[r, currentcolumn] / matrix[currentrow, currentcolumn]
            replacement(row, currentrow, r, -zarib, matrix)


def make_pivot_one(row, column, matrix):
    pivots = find_pivots(0, row, column, matrix)
    for p in range(len(pivots)):
        num = 1 / round(matrix[p, pivots[p]], 1)
        scaling(row, p, num, matrix)
        # print_matrix(matrix)
    return matrix


def find_pivot_in_row(row, column, matrix):
    pivot_rows = []
    for r in range(row):
        if iszero(r, column, matrix):
            continue
        for c in range(column):
            if matrix[r, c] != 0:
                pivot_rows.append(r)
                break
    return pivot_rows


def make_reduced_echelon_form(row, column, matrix):
    res = echelon_form(row, column, matrix)
    print_matrix(res)
    pivot_columns = find_pivots(0, row, column, matrix)
    pivot_rows = find_pivot_in_row(row, column, matrix)
    for p in range(len(pivot_rows) - 1, -1, -1):
        zero_upper_rows(pivot_rows[p], row, pivot_columns[p], res)
    make_pivot_one(row, column, res)
    print(" >> REDUCED ECHELON FORM << ")
    print_matrix(res)
    print("*****************************************")
    return res


# FINDING THE ANSWERS OF THE SYSTEM
def is_consistent(matrix):
    for row in range(matrix.shape[0]):
        coefficients_are_zero = True
        for col in range(matrix.shape[1]-1):
            if matrix[row][col] != 0:
                coefficients_are_zero = False
                break
        if coefficients_are_zero and matrix[row][-1] != 0:
            return True
    return False


def one_answer(row, column, matrix):  # return true is it has one answer - false in it has 0 or infinite answers
    pivot_columns = find_pivots(0, row, column, matrix)
    if len(pivot_columns) == column:
        return True
    return False


def pivot_in_a_row(currentrow, column, matrix):
    pivot = 0
    for col in range(column):
        if abs(round(matrix[currentrow][col], 1)) == 0.0:
            pivot += 1
        else:
            break
    return pivot


def infinite_answer(row, column, matrix):
    for r in range(row):
        pivot = pivot_in_a_row(r, column, matrix)
        if pivot != column:
            print('    x', end='')
            print(pivot + 1, '= ', end='')
            for col in range(column):
                if matrix[r][col] != 0 and col != pivot:
                    print('- x', end='')
                    print(col + 1, end='')
                    print('(', end='')
                    print(round(matrix[r][col], 1), end=')\t')
            print('+', round(matrix[r][-1], 1))

    for i in range(column):
        if i not in find_pivots(0, row, column, matrix):
            print('    x', end='')
            print(i + 1, '= free variable')


def vector_answer(totalrow, totalcolumn, matrix):
    var_num = matrix.shape[1] - 1
    if var_num > matrix.shape[0]:
        zero_row = []
        for i in range(matrix.shape[1]):
            zero_row.append(0.0)
        zero_row = np.array(zero_row)
    for i in range(var_num - matrix.shape[0]):
        matrix = np.vstack((matrix, zero_row))
    x = []
    for i in range(0, var_num):
        x.append(i + 1)
    x = np.array(x)
    x = x.reshape(-1, 1)

    c = []
    for i in range(0, var_num):
        if i < totalrow:
            c.append(matrix[i][-1])
        else:
            c.append(0.)
    c = np.array(c)
    c = c.reshape(-1, 1)
    v = {}
    for i in range(0, var_num):
        if i not in find_pivots(0, totalrow, totalcolumn, matrix):
            arr = []
            for var in range(0, var_num):
                if var != i:
                    arr.append(-matrix[var][i])
                else:
                    arr.append(1.0)

            arr = np.array(arr)
            arr = arr.reshape(-1, 1)
            v[i] = arr
    print("><><><><><><><><><")
    print(' X:')
    for r in range(x.shape[0]):
        if r == 0:
            print('[[', end=' ')
        else:
            print(' [', end=' ')
        for col in range(x.shape[1]):
            print('x', end='')
            print(round(x[r][col], 1), end='\t')
        if r == x.shape[0] - 1:
            print(']]')
        else:
            print(']')

    print("><><><><><><><><><")
    print('C: (Constant Vector)')
    print_matrix(c)
    print("><><><><><><><><><")
    for j in v:
        print('V', end='')
        print(j + 1, ': (Free Variable Vector)')
        print_matrix(v[j])
        print("><><><><><><><><><")
    print("The 'vector equation' of the system is :")
    print('  X = C ', end='')
    for y in v:
        print('+ x', end='')
        print(y + 1, end='')
        print('(V', end='')
        print(y + 1, end=') ')
    print("\n")


def solution_set(row, column, reduced_echelon_form):  # the solution set of system
    if is_consistent(reduced_echelon_form):
        print("INCONSISTENT")
        return
    if one_answer(row, column, reduced_echelon_form):  # if system has only one answer
        answer = []
        for r in range(reduced_echelon_form.shape[0]):
            answer.append(round(reduced_echelon_form[r, reduced_echelon_form.shape[1] - 1], 1))
        final = np.array(answer).reshape(column, 1)
        print(" The system has one answer")
        print("The answer is :")
        print_matrix(final)
    else:  # system has many answers
        print("System has infinite answers")
        print("The answer is : ")
        infinite_answer(row, column, reduced_echelon_form)
        vector_answer(row, column, reduced_echelon_form)


# EXERCISE 1
# create a m*n matrix with input
# m = rows and n = columns
print(" >>> EXERCISE 1 <<< ")
row = int(input("Enter the number of rows:"))
column = int(input("Enter the number of columns:"))

print("Enter the entries in a single line (separated by SPACE): ")
print("Enter all elements of first row (left to right) then move to next row")

entries = list(map(float, input().split()))  # coefficients

print("enter the constants(separated by space) - just like the coefficients: ")
constants = list(map(float, input().split()))

coefficientMatrix = np.array(entries).reshape(row, column)  # coefficients
constantMatrix = np.array(constants).reshape(row, 1)  # constant column
augmented = np.c_[coefficientMatrix, constantMatrix]

# initial augmented matrix as an output
print("Initial matrix :")
print_matrix(coefficientMatrix)
print("Coefficients :")
print_matrix(constantMatrix)
print("Initial AUGMENTED matrix :")
print_matrix(augmented)
reduced_echelon_form = make_reduced_echelon_form(row, column, augmented)
solution_set(row, column, reduced_echelon_form)
# ########################
# ########################
# ########################
# EXERCISE 2
print("\n>>> EXERCISE 2 <<< \n")
print("Matrix A :")
print_matrix(coefficientMatrix)


def calculte_dim_null(row, column, matrix):
    pivot_columns = find_pivots(0, row, column, matrix)
    dim_nul = column - len(pivot_columns)
    return dim_nul


def calculate_rank(row, column, matrix):
    dim_nul = calculte_dim_null(row, column, matrix)
    rank = column - dim_nul
    print("Rank of A :")
    print(rank)
    return rank


dim_null = calculte_dim_null(row, column, reduced_echelon_form)
print("Dim NullA :")
print(dim_null)
calculate_rank(row, column, reduced_echelon_form)
