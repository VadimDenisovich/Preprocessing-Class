import pandas as pd 
import numpy as np 
import warnings 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import zscore 
import tensorflow as tf
import torch

class Dataset: 
    def __init__(self, data: pd.DataFrame): 
        self.data = data
        self.encoded_data = None # Закодированные категориальные признаки 

    # Приватный метод для удаления None строк 
    def __clear_all_empty_strings(self):
        # Удаляем все пустые (None) строки
        # Если строка состоит исключительно из None, то удаляем 
        self.data = self.data.dropna(how='all')  

    # Конвертируем строковый столбец, содержащий в строках только числа в вещественные(float) числа
    def __convert_string_nums_to_float(self):
        for col in self.data.columns: 
            # Проверяем является ли столбец строковым типом
            if self.data[col].dtype == object: 
                try: 
                    # Если столбец полностью состоит из чисел, которые записаны 
                    # в качестве строк, то переводим его во float 
                    self.data[col] = self.data[col].astype(float)
                except ValueError: 
                    pass

    # Если в числовом столбце есть None(NaN), то заменяем его на медианое 
    def __convert_nan_to_median(self):
        # Берём название числовых столбцов
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        median_values = self.data[numeric_columns].median()
        # fillna параметр inplace = True позволяет изменять текущую таблицу, не создавая её копию 
        self.data[numeric_columns] = self.data[numeric_columns].fillna(median_values) 

    # Удаление выбросов (или, проще говоря, крайне больших чисел)
    def __clear_emissions(self, clear_emissions_type='z'):
        numeric_data = self.data.select_dtypes(include=[np.number])
        match clear_emissions_type: 
            case 'z': 
                # Формируем Z-оценки 
                # Z-оценка: насколько стандартных отклонений значение отклоняется от среднего
                z_scores = pd.DataFrame(np.abs(zscore(numeric_data)), columns=numeric_data.columns)
                # Удаляем строки, где есть выбросы.  
                self.data = self.data[(z_scores < 3).all(axis=1)]
                # Общепринятый критерий: 
                # Когда Z-оценка для значения больше 3, то оно считается выбросом.
            case 'iqr': 
                # Формируем межквартильный размах (IQR) 
                # IQR = Q3 - Q1
                # q1 - число равное первой четверти 
                # q3 - число третьей четверти 
                # Находим верхние границы [q1 - 1.5 * IQR, q3 + 1.5 * IQR] 
                # Если число выходит за границы, то это выброс
                numeric_columns = numeric_data.columns
                for col in numeric_columns: 
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR 
                    upper = Q3 + 1.5 * IQR 

                    # Оставляем только данные в границах 
                    # Как работает? Мы проверяем значения по столбцу на вхождение в границы,
                    # если не входит, то удаляется вся строчка 
                    self.data = self.data[(self.data[col] < upper) & (self.data[col] > lower)]

    # Приватный метод для кодирования категориальных прицнипов (нужно, чтобы упростить дальнейшую работу анализа данных)
    def __encode_categorical_features(self, encoding_type='onehot'):
        # Записываем все названия столбцов содержащие категориальные признаки 
        categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        if categorical_features: 
            print(f'Категориальные признаки: {categorical_features}')

            # Изменяем представление категориальных данных для удобства работы алгоритмов ml
            if encoding_type == 'onehot': 
                # OneHot: представление данных в виде бинарных векторов
                # sparse=False возвращает результат в виде np.darray (обычная плотная матрица)
                # эффективнее было sparse=True, поскольку у нас большая часть нули в векторе, но 
                # pandas не умеет работать c CSR разреженными матрицами 
                encoder = OneHotEncoder(sparse_output=False)
                # fit_transform - вначале fit обучается на данных, а потом сразу трансформирует их.
                # в columns мы присваиваем новые столбцы, относительно всех вариаций для какого-то конкретного 
                # категориального признака. 
                # Например, категорильный признак "цвет" заменится столбцы на "синий", "красный", "зелёный"
                self.encoded_data = pd.DataFrame(encoder.fit_transform(self.data[categorical_features]),
                                                columns=encoder.get_feature_names_out(categorical_features))
            elif encoding_type == 'label': 
                # Label: присваивает каждому категориальному признаку уникальный номер 
                encoder = LabelEncoder()
                # Изменяем столбцы, содержащие категориальные признаки 
                # .apply() применяется к каждому столбцу
                # .fit_transform() обучается на каждому отдельном столбце, а затем сразу 
                # преобразует категорию в числовую метку 
                self.encoded_data = self.data[categorical_features].apply(encoder.fit_transform)
            else: 
                warning.warn('Неподдерживаемый тип кодирования.')
                
        else: 
            print('Категориальные признаки не обнаружены')

    # Приватный метод для соединения encoded_data + self.data, состоящая из столбцов
    def __concat_data(self):
        categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        # Преобразовываем данные, соединяя числовые столбцы с self.data 
        # с закодированными категориальными столбцами
        # axis = 0 (конкатенация по строкам) 
        # axis = 1 (конкатенация по столбцам) 
        self.data = pd.concat([self.data.drop(columns=categorical_features), self.encoded_data], axis=1)
    
    def preparing(self, clear_emissions_type='z', encoding_type='onehot'): 
        # Чистим данные, несоответствующие формату
        self.__clear_all_empty_strings()

        # Если есть числовые столбцы, представленные в виде строк, то конвертируем
        self.__convert_string_nums_to_float() 
        
        # Заменяем NaN на медианное значение (только для чисел) в столбце 
        self.__convert_nan_to_median()

        # Удаление выбросов в числовых столбцах 
        self.__clear_emissions(clear_emissions_type) 
                    
        # TODO: cтандартизация и нормализация числовых значений, если требуется. 
        # Комментарий: потом, мне ща лень пиздец)))))))))) (не забыть удалить)
        
        # Кодируем категориальные принципы
        self.__encode_categorical_features(encoding_type)
        
        # Соединяем закодированные данные с числовыми столбцами 
        self.__concat_data()

        # Список всех фич для display метода 
        features = self.data.columns.tolist()
        print('Features: ', features)

    # Визуализация
    def display(self, feature=None): 
        # Если указано какой-то конкретный столбец
        if feature: 
            if feature not in self.data.columns: 
                print(f'Признак {feature} не найден в данных.')
                return 
            print(self.data[feature].describe())
            # Создаем гистограмму, где на оси X данные из фичи, а ось Y отвечает за частоту их появляения в столбце 
            self.data[feature].hist() 
            plt.title(f'Гистограмма для {feature}')
            plt.show()
        else: 
            print(self.data.describe())
            # figsize задает размер фигуры в дюймах: 10 длина, 8 ширина 
            self.data.hist(figsize=(10, 8))
            plt.show()

    # Кросс-валидация метод оценки модели 
    def prepare_for_cross_validation(self, n_splits=5, stratify_by=None):
        # n_splits - кол-во складок(folds) или частей, на которые будет разделены данные
        # stratify_by - это указание какого-то конкретного признака 
        # (категориального, непрерывного, искусственного и т.д) 
        # признак указывается, чтобы сохранить пропорции в каждой тестовой и обучающей частях данных
        # какого-то класса, который редко представлен в этом признаке. 
        # класс в данном контексте значит какое-то уникальное значение в столбце. 
        
        if stratify_by and stratify_by in self.data.columns:
            # StratifiedKFold кросс-валидация 
            # Стратификация данных с сохранением пропорций 
            # Метод разделяет данные на k складок таким образом, чтобы пропорции классов в каждой складке 
            # соответствовали пропорциям классов в исходном наборе данных.
            # Работает так же, как и KFold (описание ниже)
            # random_state=42 - обеспечивает одинаковое разбиение на части 
            # это нужно для будущего тестирования и отладки модели, чтобы быть точно уверенным
            # что меняется модель, а не какие-то другие факторы
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = skf.split(self.data, self.data[stratify_by])
        else: 
            # Kfold кросс-валидация 
            # Мы делим данные на k частей 
            # Берём из них k-1 частей для обучения и оставшуюся для тестирования
            # Так проделываем k раз, чтобы модель протестировалась на всех частях
            kf = Kfold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = kf.split(self.data)

        # self.data.iloc[train_idx] - выбирает строки для обучения
        # self.data.iloc[test_idx] - выбирает строки для обучения 
        # Возвращает список кортежей из обучающего и тестовых частей данных 
        return [(self.data.iloc[train_idx], self.data.iloc[test_idx]) for train_idx, test_idx in splits]

    # Преобразуем self.data в NumPy массив
    def __conver_to_numpy(self):
        return self.data.to_numpy()

    # Преобразуем NumPy массив в тензор tensorflow
    def __convert_to_tensorflow(self):
        return tf.convert_to_tensor(self.data.to_numpy())

    # Преобразуем NumPy массив в тензор torch
    def __convert_to_torch(self):
        return torch.tensor(self.data.to_numpy())
    
    def transform(self, library='numpy'): 
        match library: 
            case 'numpy':
                self.__convert_to_numpy()
            case 'tensorflow':
                self.__convert_to_tensorflow()
            case 'pytorch':
                self.__convert_to_torch() 
            case _:
                pass

data = pd.DataFrame({
    "form": ['circle', 'rectangle', 'square', 'rhombus', 'circle', 'rectangle', 'square', 'rhombus'],
    "color": ['red', 'purple', None, 'violet', 'purple', 'white', 'black', 'yellow'],
    "area": [10, 10, 15, 24, 39, 9, 1000000, 23],
    "priority": [3, 1, 2, 4, None, 1, 2, 4],
})

dataset = Dataset(data) 
dataset.preparing(clear_emissions_type='z', encoding_type='label') # использую iqr, потому что Z-оценка оказался не чувствительной к выбросам в моей выборке
dataset.display()
print(dataset.prepare_for_cross_validation(n_splits=2, stratify_by='area')) # поставил разделение на 2 части, потому что данные невелики
dataset.transform(library='tensorflow') 
