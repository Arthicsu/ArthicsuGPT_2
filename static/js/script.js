function addClassToInput(className) {
    const input = document.getElementById('selected_classes');
    const currentValue = input.value.trim();

    if (currentValue === '') {
        input.value = className;
    } else {
        // Проверяем, нет ли уже этого класса
        const classes = currentValue.split(',').map(c => c.trim());
        if (!classes.includes(className)) {
            input.value = currentValue + ', ' + className;
        }
    }
}

// Подсказка при фокусе на поле классов
document.getElementById('selected_classes').addEventListener('focus', function() {
    this.placeholder = 'person, car, dog, cat, chair, etc...';
});

document.getElementById('selected_classes').addEventListener('blur', function() {
    this.placeholder = 'Например: person, car, dog, cat';
});