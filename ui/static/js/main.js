async function classifyTask() {
    const taskDescription = document.getElementById('taskDescription').value;
    const resultText = document.getElementById('resultText');
    const categoryResult = document.getElementById('categoryResult');
    
    if (!taskDescription) {
        alert('Please enter a task description');
        return;
    }
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: taskDescription })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            resultText.textContent = `Input Text: ${data.text}`;
            categoryResult.textContent = `Predicted Category: ${data.category}`;
        } else {
            throw new Error(data.error || 'Classification failed');
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}