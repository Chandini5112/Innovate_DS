document.addEventListener('DOMContentLoaded', () => {
    const charElement = document.getElementById('chart');
    if(!charElement){
        console.error('Canvas Element nt found.');
        return;
    }
    const ctx = charElement.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 100);
    gradient.addColorStop(0, 'rgba(250, 0, 0, 1)');
    gradient.addColorStop(1, 'rgba(136, 255, 0, 1)');


    const forecastItems = document.querySelectorAll('.forecast');

    const temps = [];
    const times = [];
    forecastItems.forEach(item => {
    const time = item.querySelector('.time')?.textContent;
    const temp = item.querySelector('.temp')?.textContent;

    if (time && temp) {
        times.push(time.trim());
        temps.push(parseFloat(temp));
    }
});

    if (temps.length === 0 || times.length === 0) {
        console.error('Temp or time values are missing.');
        return;
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [
                {
                    label: 'Celsius Degrees',
                    data: temps,
                    borderColor: gradient,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 2,
                },
            ],
        },
        options: {
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
                y: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
            },
            animation: {
                duration: 750,
            },
        },
    });
});