# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv as inv

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciąg wejściowy Nx1
    :param y: ciąg wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
     uzyskanymi z wielowamiu o parametrach w dla wejść x
    """
    N = x.size
    return np.sum(np.square(np.subtract(y, polynomial(x, w)))) / N


def design_matrix(x_train, M):
    """
    :param x_train: ciąg treningowy Nx1
    :param M: stopień wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """
    N = x_train.size
    des_matrix = np.zeros(shape=(N, M + 1))
    for n in range(N):
        for m in range(M + 1):
            des_matrix[n][m] = x_train[n] ** m
    return des_matrix


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego 
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """
    des_matrix = design_matrix(x_train, M)
    w = inv(des_matrix.transpose() @ des_matrix) @ des_matrix.transpose() @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return w, err


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd 
    średniokwadratowy dopasowania
    """
    des_matrix = design_matrix(x_train, M)
    w = inv(des_matrix.transpose() @ des_matrix + regularization_lambda * np.eye(M + 1)) @ des_matrix.transpose() @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return w, err


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na 
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na 
    ciągach treningowym i walidacyjnym
    """
    train_err = []
    val_err = []
    w_list = []
    for m in M_values:
        (w, err) = least_squares(x_train, y_train, m)
        w_list.append(w)
        train_err.append(err)
        val_err.append(mean_squared_error(x_val, y_val, w))
    index = val_err.index(min(val_err))
    return w_list[index], train_err[index], val_err[index]


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    train_err = []
    val_err = []
    w_list = []
    for l in lambda_values:
        (w, err) = regularized_least_squares(x_train, y_train, M, l)
        w_list.append(w)
        train_err.append(err)
        val_err.append(mean_squared_error(x_val, y_val, w))
    index = val_err.index(min(val_err))
    return w_list[index], train_err[index], val_err[index], lambda_values[index]
