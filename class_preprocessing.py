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
    def __init__(self): 
        self.__data = None
        self.__encoded_data = None # Закодированные категориальные признаки 
        self.__categorical_features = None # Список категориальных признаков

    # getter для __data 
    @property
    def data(self): 
        return self.__data

    # setter для __data 
    @data.setter
    def data(self, data):
        self.__data = data 
        print('Dataset is updated!')

    # Собираем строковые категориальные признаки 
    def __set_categorical_features(self, less_then=None): 
        self.__categorical_features = self.__data.select_dtypes(include=[object]).columns.tolist() 
        if less_then: 
            # self.__data.nunique() < less_then - берёт кол-во уникальных значений в столбце и сравнивает их число, которое мы задали
            # напоминаю, less_then это число, которое показывает меньше скольки уникальных значений непрерывный признак 
            # начинает считаться категориальным. Если меньше 10, например, то признак становится категориальным. 
            selected_columns = self.__data.columns[self.__data.nunique() < less_then].tolist()
            self.__categorical_features += selected_columns
            self.__categorical_features = list(set(self.__categorical_features))
        print("Категориальные признаки:", self.__categorical_features)
    
    # Приватный метод для удаления None строк 
    def __clear_all_empty_strings(self):
        # Удаляем все пустые (None) строки
        # Если строка состоит исключительно из None, то удаляем 
        self.__data = self.__data.dropna(how='all')  

    # Конвертируем строковый столбец, содержащий в строках только числа в вещественные(float) числа
    def __convert_string_nums_to_float(self):
        for col in self.data.columns: 
            # Проверяем является ли столбец строковым типом
            if self.__data[col].dtype == object: 
                try: 
                    # Если столбец полностью состоит из чисел, которые записаны 
                    # в качестве строк, то переводим его во float 
                    self.__data[col] = self.__data[col].astype(float)
                except ValueError: 
                    pass

    # Если в числовом столбце есть None(NaN), то заменяем его на то, что выбрал пользователь или удаляем  
    # 'median' - заменяет на медиану
    # 'mode' - заменяет на самое частое 
    # 'medium' - на среднее значение
    # 'const' - заменяем на заданную нами константу (учтите, что оно меняет во всех столбцах и где-то значение может не подойти)
    def __change_nan_to_something(self, change_to=None):
        # Берём название числовых столбцов
        numeric_columns = self.__data.select_dtypes(include=[np.number]).columns

        # Если задано, на что заменять, то мы меняем на это 
        if change_to:
            global changed
            match change_to:
                case 'median':
                    # Получаем медиану по столбцу 
                    changed = self.__data[numeric_columns].median()
                case 'mode':
                    # Получаем самое частое (моду) по столбцу
                    # Метод .iloc[0] извлекает первую строку из полученного DataFrame с модами
                    changed = self.__data[numeric_columns].mode().iloc[0]
                case 'mean':
                    changed = self.__data[numeric_columns].mean(skipna=True)
                case 'const': 
                    const = int(input('print value which replace NaN: ')) 
                    changed = self.__data[numeric_columns].fillna(const)
                case _: 
                    raise ValueError(f"Unknown change_to value: {change_to}")
                    
            # fillna параметр inplace = True позволяет изменять текущую таблицу, не создавая её копию 
            self.__data[numeric_columns] = self.__data[numeric_columns].fillna(changed) 

        # Если не задано на что заменять, то мы удаляем строки с None
        else: 
            self.__data = self.__data.dropna(subset=numeric_columns)

            

    # Если в категориальном столбце есть None(NaN), то заменяем его модой (наиболее частым значением)
    def __convert_nan_to_mode(self): 
        # Столбцы с категориальными признаками 
        string_categorical_features = self.__data.select_dtypes(include=['object']).columns.tolist() 
        for col in string_categorical_features: 
            # Самое частое слово 
            mode_word = self.data[col].mode()[0]
            # fillna заменяет в каждом столбце самое часто встречаемое слово
            self.__data[col] = self.__data[col].fillna(mode_word)

    # Удаление выбросов (или, проще говоря, крайне больших чисел)
    def __clear_emissions(self, clear_emissions_type='z'):
        numeric_data = self.__data.select_dtypes(include=[np.number])
        match clear_emissions_type: 
            case 'z': 
                # Формируем Z-оценки 
                # Z-оценка: насколько стандартных отклонений значение отклоняется от среднего
                z_scores = pd.DataFrame(np.abs(zscore(numeric_data)), columns=numeric_data.columns)
                # Удаляем строки, где есть выбросы.  
                self.__data = self.__data[(z_scores < 3).all(axis=1)]
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
                    Q1 = self.__data[col].quantile(0.25)
                    Q3 = self.__data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR 
                    upper = Q3 + 1.5 * IQR 

                    # Оставляем только данные в границах 
                    # Как работает? Мы проверяем значения по столбцу на вхождение в границы,
                    # если не входит, то удаляется вся строчка 
                    self.__data = self.__data[(self.data[col] < upper) & (self.data[col] > lower)]

    # Приватный метод для кодирования категориальных прицнипов (нужно, чтобы упростить дальнейшую работу анализа данных)
    def __encode_categorical_features(self, encoding_type='onehot'):
        # Записываем все названия столбцов содержащие категориальные признаки 
        string_categorical_features = self.__data.select_dtypes(include=['object']).columns.tolist()
        if string_categorical_features: 
            print(f'Категориальные признаки, которые будут декодированы: {string_categorical_features}')

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
                self.__encoded_data = pd.DataFrame(encoder.fit_transform(self.data[string_categorical_features]),
                                                columns=encoder.get_feature_names_out(string_categorical_features))
            elif encoding_type == 'label': 
                # Label: присваивает каждому категориальному признаку уникальный номер 
                encoder = LabelEncoder()
                # Изменяем столбцы, содержащие категориальные признаки 
                # .apply() применяется к каждому столбцу
                # .fit_transform() обучается на каждому отдельном столбце, а затем сразу 
                # преобразует категорию в числовую метку 
                self.__encoded_data = self.__data[string_categorical_features].apply(encoder.fit_transform)
            else: 
                warning.warn('Неподдерживаемый тип кодирования.')
                
        else: 
            print('Категориальные признаки не обнаружены')

    # Приватный метод для соединения encoded_data + self.data, состоящая из столбцов
    def __concat_data(self):
        string_categorical_features = self.__data.select_dtypes(include=['object']).columns.tolist()
        # Преобразовываем данные, соединяя числовые столбцы с self.data 
        # с закодированными категориальными столбцами
        # axis = 0 (конкатенация по строкам) 
        # axis = 1 (конкатенация по столбцам) 
        self.__data = pd.concat([self.__data.drop(columns=string_categorical_features), self.__encoded_data], axis=1)

    '''
    @nan_in_nums_cols_change_to - отвечает за то, на какое значения мы заменяем NaN в числовых столбцах 
        median - установит медианное значение 
        mean - установит среднее знаничение
        mode - установит моду 
        const - ставит константу 
        по умолчанию - None, очищается строка, где встречается NaN 
    @condition_to_categorical_features - указывает число, показывающее кол-во уникальных значений. 
        Если столбец содержит меньше уникальных значений - то это категориальный признак. 
        Eсли больше, то это непрерывный признак. 
        по умолчанию None - будут закинуты только строковые столбцы, как категориальные
    @clear_emissions_type - указывает метод, каким мы будем определять выбросы 
        z - Z-оценка
        iqr - межквартильный размах 
        по умолчанию - z 
    @encoding_type - способ кодирования категориальных признаков 
        onehot - OneHotEncoder() 
        label - LabelEncoder() 
        по умолчанию - onehot
    '''
    def preparing(self, clear_emissions_type='z', encoding_type='onehot', nan_in_nums_cols_change_to=None, condition_to_categorical_features=None): 
        # Чистим данные, несоответствующие формату
        self.__clear_all_empty_strings()

        # Если есть числовые столбцы, представленные в виде строк, то конвертируем
        self.__convert_string_nums_to_float() 

        # Заменяем NaN на какое-то значение (только для чисел) в столбце. Например, медиану
        self.__change_nan_to_something(change_to=nan_in_nums_cols_change_to)

        # Удаление выбросов в числовых столбцах 
        self.__clear_emissions(clear_emissions_type) 
                    
        # TODO: cтандартизация и нормализация числовых значений, если требуется. 
        
        # Заменяем NaN на моду 
        self.__convert_nan_to_mode()

        # Собираем список категориальных признаков, перед кодированием
        self.__set_categorical_features(less_then=condition_to_categorical_features) 
        
        # Кодируем категориальные принципы
        self.__encode_categorical_features(encoding_type)
        
        # Соединяем закодированные данные с числовыми столбцами 
        self.__concat_data()

        # Список всех фич для display метода 
        features = self.__data.columns.tolist()
        print('Features: ', features)

    # Визуализация
    def display(self, feature=None): 
        # Если указано какой-то конкретный столбец
        if feature: 
            if feature not in self.__data.columns: 
                print(f'Признак {feature} не найден в данных.')
                return 
            print(self.__data[feature].describe())
            # Создаем гистограмму, где на оси X данные из фичи, а ось Y отвечает за частоту их появляения в столбце 
            self.__data[feature].hist() 
            plt.title(f'Гистограмма для {feature}')
            plt.show()
        else: 
            print(self.__data.describe())
            # figsize задает размер фигуры в дюймах: 10 длина, 8 ширина 
            self.__data.hist(figsize=(10, 8))
            plt.show()
    
    # Кросс-валидация метод оценки модели 
    def prepare_for_cross_validation(self, n_splits=5, stratify_by=None):
        # n_splits - кол-во складок(folds) или частей, на которые будет разделены данные
        # stratify_by - это указание какого-то конкретного признака 
        # (категориального, непрерывного, искусственного и т.д) 
        # признак указывается, чтобы сохранить пропорции в каждой тестовой и обучающей частях данных
        # какого-то класса, который редко представлен в этом признаке. 
        # класс в данном контексте значит какое-то уникальное значение в столбце. 
        if stratify_by:
            # StratifiedKFold кросс-валидация 
            # Стратификация данных с сохранением пропорций 
            # Метод разделяет данные на k складок таким образом, чтобы пропорции классов в каждой складке 
            # соответствовали пропорциям классов в исходном наборе данных.
            # Работает так же, как и KFold (описание ниже)
            # это нужно для будущего тестирования и отладки модели, чтобы быть точно уверенным
            # что меняется модель, а не какие-то другие факторы

            # если нужно стратифицировать по нескольким признакам 
            if isinstance(stratify_by, list) and all(feature in self.__data.columns for feature in stratify_by):  
                stratify_cols = []
                for col in stratify_by: 
                    if col in self.__categorical_features: 
                        stratify_cols.append(self.__data[col].astype(str))
                    else: 
                        # Бинируем непрерывные признаки 
                        # Бинирование — это процесс преобразования непрерывного числового признака в категориальный путем 
                        # разделения его диапазона значений на несколько интервалов, называемых бинами 
                        self.__data[f'{col}_binned_feature'] = pd.qcut(self.data[col], q=4, labels=False)
                        # Преобразовываем значения бинированного столбца в строки и добавляем в stratify_cols
                        stratify_cols.append(self.__data[f'{col}_binned_feature'].astype(str))
                self.__data['stratify_feature'] = stratify_cols[0]
                for col in stratify_cols[1:]:
                    self.__data['stratify_feature'] += '_' + col
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                splits = skf.split(self.__data, self.__data['stratify_feature'])
                return [(self.__data.iloc[train_idx], self.__data.iloc[test_idx]) for train_idx, test_idx in splits] 

            # если на передан одиночный признак
            elif stratify_by in self.__data.columns: 
                if stratify_by in self.__categorical_data: 
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    splits = skf.split(self.__data, self.__data[stratify_by])
                    # self.data.iloc[train_idx] - выбирает строки для обучения
                    # self.data.iloc[test_idx] - выбирает строки для обучения 
                    # Возвращает список кортежей из обучающего и тестовых частей данных 
                    return [(self.__data.iloc[train_idx], self.__data.iloc[test_idx]) for train_idx, test_idx in splits]
                
                # если признак непрервный
                else:
                    # создаем новый столбец, он представляет из себя категориальный признак, переведенный из непрерывного
                    # представим, что расположили данные на отрезке, поделим отрезок на 4 части, и в столбец относим число от 1 до 4, к какому интервалу 
                    # относится конкретный непрерывный аргумент 
                    self.__data[f'{stratify_by}_binned_feature'] = pd.qcut(self.data[stratify_by], q=4, labels=False)
                    # random_state=42 - обеспечивает одинаковое разбиение на части 
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    splits = skf.split(self.__data, self.__data[f'{stratify_by}_binned_feature'])
                    return [(self.__data.iloc[train_idx], self.__data.iloc[test_idx]) for train_idx, test_idx in splits]
        else: 
            # Kfold кросс-валидация 
            # Мы делим данные на k частей 
            # Берём из них k-1 частей для обучения и оставшуюся для тестирования
            # Так проделываем k раз, чтобы модель протестировалась на всех частях
            kf = Kfold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = kf.split(self.data)
            return [(self.__data.iloc[train_idx], self.__data.iloc[test_idx]) for train_idx, test_idx in splits]

    # Преобразуем self.data в NumPy массив
    def __conver_to_numpy(self):
        return self.__data.to_numpy()

    # Преобразуем NumPy массив в тензор tensorflow
    def __convert_to_tensorflow(self):
        # .apply() применяется ко всем столбцам 
        # pd.to_numeric пытается преобразовать каждое значение в числовой формат (например, int или float)
        # errors='coerce' позволяет заменить на NaN те значения, что не смогли замениться на число 
        # .fillna(0) заменяет NaN на ноль  
        self.__data = self.__data.apply(pd.to_numeric, errors='coerce').fillna(0)
        return tf.convert_to_tensor(self.__data.to_numpy())

    # Преобразуем NumPy массив в тензор torch
    def __convert_to_torch(self):
        self.__data = self.__data.apply(pd.to_numeric, errors='coerce').fillna(0)
        return torch.tensor(self.__data.to_numpy())
    
    def transform(self, library='numpy'): 
        match library: 
            case 'numpy':
                self.__convert_to_numpy()
                print('Succesfully converted to numpy!')
            case 'tensorflow':
                self.__convert_to_tensorflow()
                print('Succesfully converted to tensorflow!')
            case 'pytorch':
                self.__convert_to_torch() 
                print('Succesfully converted to pytorch!')
            case _:
                pass

example_data = pd.DataFrame({
    "form": ['circle', 'rectangle', 'square', None, 'circle', 'rectangle', 'square', 'rhombus', 'circle', 'rhombus', 'rectangle', 'rhombus'],
    "color": ['red', 'purple', None, 'violet', 'purple', 'white', 'black', 'yellow', 'purple', 'red', 'green', 'green'],
    "area": [10, 11, 15, 24, 39, 9, 1000000, 23, 1000, 11, 14, 22],
    "priority": [3, np.nan, 2, 4, None, 1, 2, 4, 5, 3, 5, 1],
    "names": ['V', 'T', 'V', 'U', 'T', 'V', 'U', 'U', 'G', 'E', 'T', 'E']
})

dataset = Dataset()
dataset.data = example_data

# использую iqr, потому что Z-оценка оказался не чувствительной к выбросам в моей выборке 
# nan_in_nums_cols_change_to - отвечает за то, на какое значения мы заменяем NaN в числовых столбцах 
# condition_to_categorical_features - указывает число, показывающее кол-во уникальных значений. Если столбец содержит меньше уникальных значений - 
# то это категориальный признак, если больше, то это непрерывный признак. 
dataset.preparing(clear_emissions_type='iqr', encoding_type='label', nan_in_nums_cols_change_to='median', condition_to_categorical_features=6)


dataset.display()

print(dataset.prepare_for_cross_validation(n_splits=2, stratify_by=['color', 'names'])) # поставил разделение на 2 части, потому что данные невелики

dataset.transform(library='pytorch') 
