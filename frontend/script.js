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

    let myStockChart = null;

    predictButton.addEventListener('click', () => {
        const ticker = tickerInput.value.trim().toUpperCase();
        console.log("1. Button clicked for:", ticker); // DEBUG LOG

        if (!ticker) {
            alert('Please enter a stock ticker.');
            return;
        }

        // Reset UI state
        loader.classList.remove('hidden');
        dashboardContainer.classList.add('hidden');
        resultContainer.innerHTML = '';
        forecastListEl.innerHTML = ''; 

        // --- IMPORTANT: Verify this URL matches your Render service ---
        fetch('https://ai-stock-predictor-n85s.onrender.com/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker }),
        })
        .then(response => {
            console.log("2. Response received. Status:", response.status); // DEBUG LOG
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Something went wrong');
                });
            }
            return response.json();
        })
        .then(data => {
            console.log("3. Data parsed successfully:", data); // DEBUG LOG

            // --- STEP 1: HIDE LOADER & SHOW DASHBOARD ---
            loader.classList.add('hidden');
            dashboardContainer.classList.remove('hidden');
            console.log("4. Dashboard container should be visible now."); // DEBUG LOG

            // --- STEP 2: UPDATE TEXT FIELDS ---
            companyNameEl.textContent = data.companyName;
            currentPriceEl.textContent = `$${data.currentPrice.toFixed(2)}`;
            
            // --- STEP 3: BUILD FORECAST LIST ---
            console.log("5. Building forecast list..."); 
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

            // --- STEP 4: DRAW CHART ---
            console.log("6. Drawing chart..."); 
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
            console.log("7. Chart drawn successfully."); // DEBUG LOG
        })
        .catch(error => {
            console.error("ERROR CAUGHT:", error); // DEBUG LOG
            loader.classList.add('hidden');
            dashboardContainer.classList.add('hidden');
            resultContainer.innerHTML = `<p class="result-error">Error: ${error.message}</p>`;
        });
    });
});