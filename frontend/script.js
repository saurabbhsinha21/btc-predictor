async function predict() {
  const price = document.getElementById("priceInput").value;
  const time = document.getElementById("timeInput").value;

  const response = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_price: price, target_time: time })
  });

  const result = await response.json();

  document.getElementById("result").innerHTML = `
    <h3>Prediction: ${result.prediction}</h3>
    <p>Current Price: $${result.current_price}</p>
    <p>Predicted Price: $${result.predicted_price}</p>
    <p>Target Time: ${result.target_time}</p>
    <p>Confidence: ${result.confidence}%</p>
  `;
}
