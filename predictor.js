let chart;
let lastPrices = [];

async function fetchCandles() {
  const res = await fetch('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100');
  const data = await res.json();
  return data.map(candle => ({
    close: parseFloat(candle[4]),
    volume: parseFloat(candle[5]),
    timestamp: candle[0]
  }));
}

function calculateEMA(data, period) {
  const k = 2 / (period + 1);
  let ema = data.slice(0, period).reduce((sum, v) => sum + v, 0) / period;
  let result = [ema];
  for (let i = period; i < data.length; i++) {
    ema = data[i] * k + ema * (1 - k);
    result.push(ema);
  }
  return new Array(period - 1).fill(null).concat(result);
}

function calculateRSI(closes, period = 14) {
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff >= 0) gains += diff;
    else losses -= diff;
  }
  let rs = gains / (losses || 1);
  return 100 - (100 / (1 + rs));
}

function calculateMACD(prices, short = 12, long = 26, signal = 9) {
  const emaShort = calculateEMA(prices, short);
  const emaLong = calculateEMA(prices, long);
  const macdLine = emaShort.map((v, i) => (v !== null && emaLong[i] !== null) ? v - emaLong[i] : null);
  const validMacd = macdLine.filter(v => v !== null);
  const signalLine = calculateEMA(validMacd, signal);
  return {
    macdLine,
    signalLine: new Array(macdLine.length - signalLine.length).fill(null).concat(signalLine)
  };
}

async function predict() {
  const targetPrice = parseFloat(document.getElementById("targetPrice").value);
  const targetTimeInput = document.getElementById("targetTime").value;
  const output = document.getElementById("output");

  if (!targetPrice || !targetTimeInput) {
    output.textContent = "⚠️ Please enter both target price and target time.";
    return;
  }

  const targetDate = new Date(targetTimeInput);
  const now = new Date();
  const minutesAhead = Math.floor((targetDate - now) / 60000);

  if (isNaN(targetDate.getTime()) || minutesAhead <= 0) {
    output.textContent = "❌ Invalid or past target time.";
    return;
  }

  output.textContent = "⏳ Predicting...";

  const candles = await fetchCandles();
  const prices = candles.map(c => c.close);
  const volumes = candles.map(c => c.volume);
  const timestamps = candles.map(c => new Date(c.timestamp));

  const ema = calculateEMA(prices, 10);
  const rsi = calculateRSI(prices);
  const macd = calculateMACD(prices);

  const inputs = [];
  for (let i = 0; i < prices.length; i++) {
    if (ema[i] !== null && macd.macdLine[i] !== null && macd.signalLine[i] !== null) {
      inputs.push([
        prices[i],
        ema[i],
        volumes[i],
        macd.macdLine[i],
        macd.signalLine[i]
      ]);
    }
  }

  const xs = tf.tensor2d(inputs.slice(0, -1));
  const ys = tf.tensor2d(inputs.slice(1).map(i => [i[0]]));

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 32, inputShape: [5], activation: 'relu' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  await model.fit(xs, ys, { epochs: 25, verbose: 0 });

  let lastInput = inputs[inputs.length - 1];
  for (let i = 0; i < minutesAhead; i++) {
    const next = await model.predict(tf.tensor2d([lastInput])).array();
    lastInput = [...lastInput];
    lastInput[0] = next[0][0]; // replace price with predicted price
  }

  const predictedPrice = lastInput[0];
  const diff = Math.abs(predictedPrice - targetPrice);
  const confidence = Math.max(55, Math.min(95, 80 - (diff / targetPrice) * 100));
  const direction = predictedPrice > targetPrice ? "🔼 Above" : "🔽 Below";

  output.innerHTML = `
    📊 Predicted: ${direction} <br>
    💰 Predicted Price: $${predictedPrice.toFixed(2)} <br>
    📈 EMA(10): $${ema.at(-1).toFixed(2)} <br>
    📉 RSI(14): ${rsi.toFixed(2)} <br>
    🔍 MACD: ${(macd.macdLine.at(-1) || 0).toFixed(2)} | Signal: ${(macd.signalLine.at(-1) || 0).toFixed(2)} <br>
    🎯 Confidence: ${confidence.toFixed(1)}%
  `;

  drawChart(prices, predictedPrice, timestamps);
}

function drawChart(prices, predictedPrice, timestamps) {
  const labels = timestamps.map(t => t.toLocaleTimeString().slice(0, 5));
  const predictedLabels = [...labels, 'Target'];
  const predictedData = [...prices, predictedPrice];

  if (chart) chart.destroy();

  const ctx = document.getElementById("chartCanvas").getContext("2d");
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: predictedLabels,
      datasets: [{
        label: "Price",
        data: predictedData,
        borderColor: "#007bff",
        backgroundColor: "rgba(0, 123, 255, 0.1)",
        fill: true,
        tension: 0.2
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: { display: true },
        y: { display: true }
      },
      plugins: {
        legend: { display: true }
      }
    }
  });
}
