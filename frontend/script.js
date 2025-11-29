document.addEventListener('DOMContentLoaded', () => {
    const predictButton = document.getElementById('predict-button');
    const tickerInput = document.getElementById('ticker-input');
    const loader = document.getElementById('loader');
    const dashboardContainer = document.getElementById('dashboard-container');
    const companyNameEl = document.getElementById('company-name');
    const currentPriceEl = document.getElementById('current-price');
    const resultContainer = document.getElementById('result-container');
    const chartCtx = document.getElementById('stock-chart').getContext('2d');
    const forecastListEl = document.getElementById('forecast-list');
    
    // --- Sentiment Element ---
    const sentimentBadgeEl = document.getElementById('sentiment-badge');

    let myStockChart = null;

    predictButton.addEventListener('click', () => {
        const ticker = tickerInput.value.trim().toUpperCase();

        if (!ticker) {
            alert('Please enter a stock ticker.');
            return;
        }

        // Reset UI
        loader.classList.remove('hidden');
        dashboardContainer.classList.add('hidden');
        resultContainer.innerHTML = '';
        forecastListEl.innerHTML = '';
        
        // Reset badge to loading state
        sentimentBadgeEl.className = 'badge'; 
        sentimentBadgeEl.textContent = 'Analyzing...';

        // --- REPLACE WITH YOUR RENDER URL ---
        fetch('https://ai-stock-predictor-n85s.onrender.com/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker }),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Something went wrong');
                });
            }
            return response.json();
        })
        .then(data => {
            loader.classList.add('hidden');
            dashboardContainer.classList.remove('hidden');

            companyNameEl.textContent = data.companyName;
            currentPriceEl.textContent = `$${data.currentPrice.toFixed(2)}`;
            
            // --- Update Sentiment Badge ---
            const sentiment = data.sentiment; // "Bullish", "Bearish", "Neutral"
            sentimentBadgeEl.textContent = `${sentiment} (Score: ${data.sentimentScore})`;
            
            // Reset classes
            sentimentBadgeEl.className = 'badge';
            
            // Apply color class based on sentiment
            if (sentiment === 'Bullish') {
                sentimentBadgeEl.classList.add('badge-bullish');
            } else if (sentiment === 'Bearish') {
                sentimentBadgeEl.classList.add('badge-bearish');
            } else {
                sentimentBadgeEl.classList.add('badge-neutral');
            }

            // --- Populate Forecast List ---
            data.sevenDayForecast.forEach((forecast, index) => {
                const li = document.createElement('li');
                const daySpan = document.createElement('span');
                daySpan.textContent = `Day ${index + 1} (${forecast.date}):`;
                const priceStrong = document.createElement('strong');
                priceStrong.textContent = `$${forecast.price.toFixed(2)}`;
                li.appendChild(daySpan);
                li.appendChild(priceStrong);
                forecastListEl.appendChild(li);
            });

            // --- Draw Chart with SMA ---
            if (myStockChart) {
                myStockChart.destroy();
            }
            
            myStockChart = new Chart(chartCtx, {
                type: 'line',
                data: {
                    labels: data.chartData.dates,
                    datasets: [
                        {
                            label: 'Close Price (USD)',
                            data: data.chartData.prices,
                            borderColor: '#007aff',
                            backgroundColor: 'rgba(0, 122, 255, 0.1)',
                            fill: true,
                            borderWidth: 2,
                            pointRadius: 0,
                        },
                        {
                            label: 'SMA-50 (Trend)',
                            data: data.chartData.sma50,
                            borderColor: '#ff9f40',
                            borderWidth: 2,
                            pointRadius: 0,
                            fill: false,
                            borderDash: [5, 5]
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        x: { ticks: { maxTicksLimit: 8 } },
                        y: { ticks: { callback: value => '$' + value } }
                    },
                    plugins: { legend: { display: true } }
                }
            });
        })
        .catch(error => {
            loader.classList.add('hidden');
            dashboardContainer.classList.add('hidden');
            resultContainer.innerHTML = `<p class="result-error">Error: ${error.message}</p>`;
        });
    });
});