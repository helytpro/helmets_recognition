# Recognition of helmets  

Многокомпонентный нейросетевой детектор, обеспечивающий распознавание поз людей на изображениях,
а также их классификацию по наличию/отсутствию защитного снаряжения (каски).

---

## Описание проекта  

Проект был реализован в учебных целях в рамках курса "Искуственный интеллект" НИУ МИЭТ по направлению "Прикладная математика".

В качестве исходных данных выступил набор, в котором каждому изображению, содержащему изображение объектов (людей) в касках/ без касок на предприятии, 
соответствовал json-файл, в котором была указана информация о координатах кропа (ограничивающего прямоугольника)
для каждого объекта, а также метка наличия/отсутствия каски на объекте.

Проект был реализован в три этапа:

1) Обучение классификатора на кропах из json-файлов, распознающего наличие/отсутствие каски у объекта;
2) Внедрение нейросетевого детектора людей, обнаруживающего объекты на изображениях и создающего кропы для классификатора;
3) Внедрение нейросетевого детектора позы, выделяющего на полученных кропах три точки ("голова - левое плечо - правое плечо"), по которым создается меньший ограничивающий прямоугольник для классификатора.

В качестве классификатора касок была выбрана модель **EfficientNet**, в качестве детектора людей - **YOLO11** с весами "*medium*", в качестве детектора позы - **YOLO8** с весами "*nano*". 

---

### Компоненты детектора

- *datasets.py* - содержит функции и инструменты для подготовки данных;
- *train.py* - содержит функции для обучения модели;
- *inference.py* - содержит функции для инференса модели;
- *utils.py* - содержит функции для извлечения метрик / сохранения весов модели и графиков.

---

### Результирующий пайплайн

Программные компоненты детектора были объединены в единый пайплайн 
с помощью **DVC** для упрощения процедуры инференса и внесения изменений. Также внедрение DVC позволяет сохранять наборы данных на каждом этапе эксперимента.
Пайплайн разбит на шаги, описанные в **dvc.yaml**:
- **preprocess** - предварительная обработка изображений: выделение поз на изображений, назначение меток; 
- **train** - непосредственное обучение модели;
- **evaluate** - инференс на тестовом наборе изображений.

Результатом каждого этапа становятся, соответственно, набор изображений и меток для обучения,
обученная модель и csv-файл с метриками модели на инференсе.


---

##  Результаты 

Результаты после последнего этапа:

- "Accuracy": 0.973,
- "Precision": 0.995,
- "Recall": 0.951.

Каждая из модификаций (создание кропов с помощью YOLO, поиск поз) внесла определенное повышение метрик эффективности работы модели.

---

##  Направления будущей работы (ToDo)

- внедрение инференса на видеопотоке: детекторы подобного типа должны быть адаптированы, в первую очередь, к работе с видеокамерами, осуществляющими надзор на производстве;
- оптимизация модели: файнтюнинг и квантование, внедрение новых метрик для отслеживания эффективности
- оптимизация назначения меток: на данный момент метки для инференса назначаются согласно разметке, предоставленной в json-файлах, по принципу наименьшего евклидового расстояния до центра кропа. 
Для более точного назначения может быть реализован венгерский алгоритм решения задач о назначениях;
- вынести зависимости (requirements) в отдельный список.
---

## Требуемые зависимости 
- Python  
- Git  
- dvc
- torch, sklearn, PIL, OpenCv, JSON
