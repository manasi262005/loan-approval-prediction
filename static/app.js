const API_BASE_URL = 'http://localhost:8000';

document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-links li');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            // Remove active class
            navLinks.forEach(l => l.classList.remove('active'));
            pages.forEach(p => p.classList.remove('active'));
            
            // Add active class
            link.classList.add('active');
            const targetId = link.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');

            // Load data if necessary
            if (targetId === 'data-page') loadDataExplorer();
            if (targetId === 'metrics-page') loadMetrics();
            if (targetId === 'shap-page') loadShap();
        });
    });

    // Form Submission
    const predictForm = document.getElementById('predict-form');
    predictForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(predictForm);
        const data = Object.fromEntries(formData.entries());
        
        // Convert numbers
        data.ApplicantIncome = parseFloat(data.ApplicantIncome);
        data.CoapplicantIncome = parseFloat(data.CoapplicantIncome);
        data.LoanAmount = parseFloat(data.LoanAmount);
        data.Loan_Amount_Term = parseFloat(data.Loan_Amount_Term);

        try {
            const btn = predictForm.querySelector('button');
            btn.innerHTML = 'Predicting... <span class="spinner">↻</span>';
            btn.disabled = true;

            const res = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Failed prediction');
            }

            const result = await res.json();
            showResult(result);
        } catch (err) {
            showToast(err.message);
        } finally {
            const btn = predictForm.querySelector('button');
            btn.innerHTML = 'Generate Prediction';
            btn.disabled = false;
        }
    });

    // Helper functions
    function showResult(data) {
        const container = document.getElementById('result-container');
        const badge = document.getElementById('status-badge');
        const confidenceText = document.getElementById('confidence-text');
        
        container.style.display = 'flex';
        
        // Setup badge UI
        badge.textContent = data.status.toUpperCase();
        badge.className = 'status-badge ' + (data.prediction === 1 ? 'status-approved' : 'status-rejected');
        
        // Setup confidence text
        const percent = (data.confidence * 100).toFixed(1);
        confidenceText.innerHTML = `Confidence Score: <span class="confidence-highlight">${percent}%</span>`;
        
        // Breakdown setup
        document.getElementById('detail-total-income').innerText = `₹${data.details.TotalIncome.toFixed(0)}`;
        document.getElementById('detail-emi').innerText = `₹${data.details.EMI.toFixed(2)}`;
        document.getElementById('detail-balance').innerText = `₹${data.details.BalanceIncome.toFixed(2)}`;
    }

    async function loadDataExplorer() {
        try {
            const res = await fetch(`${API_BASE_URL}/data/summary`);
            if(!res.ok) throw new Error("Could not load data stats!");
            const data = await res.json();

            document.getElementById('stat-records').innerText = data.totalRecords;
            document.getElementById('stat-features').innerText = data.totalFeatures;
            document.getElementById('stat-approved').innerText = data.approvedCount;

            const sample = data.sample;
            if(sample && sample.length > 0) {
                const headRow = document.getElementById('table-header-row');
                const tbody = document.getElementById('table-body');
                
                // Clear existing
                headRow.innerHTML = '';
                tbody.innerHTML = '';

                // Headers
                const keys = Object.keys(sample[0]);
                keys.forEach(k => {
                    const th = document.createElement('th');
                    th.innerText = k;
                    headRow.appendChild(th);
                });

                // Rows
                sample.forEach(row => {
                    const tr = document.createElement('tr');
                    keys.forEach(k => {
                        const td = document.createElement('td');
                        td.innerText = row[k] !== null ? row[k] : 'NaN';
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                });
            }
        } catch(e) {
            console.warn(e);
        }
    }

    async function loadMetrics() {
        try {
            const res = await fetch(`${API_BASE_URL}/metrics`);
            if(!res.ok) throw new Error("Metrics not found");
            const data = await res.json();

            const models = Object.keys(data.all_results);
            const accuracies = Object.values(data.all_results).map(a => parseFloat(a) * 100);

            const trace = {
                x: models,
                y: accuracies,
                type: 'bar',
                marker: {
                    color: accuracies,
                    colorscale: 'Blues',
                    showscale: true
                }
            };
            
            const layout = {
                title: false,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#94a3b8', family: 'Inter' },
                yaxis: { title: 'Accuracy (%)', range: [50, 100] }
            };

            Plotly.newPlot('metrics-chart', [trace], layout, {responsive: true});

            document.getElementById('metrics-summary').innerHTML = `
                <div class="card" style="margin-top: 2rem; border-left: 4px solid var(--success);">
                    <h3>Best Model</h3>
                    <p style="font-size: 1.2rem;"><b>${data.best_model}</b> achieving <b>${(data.accuracy*100).toFixed(2)}%</b> accuracy.</p>
                </div>
            `;
        } catch(e) {
            console.warn(e);
        }
    }

    async function loadShap() {
        try {
            const res = await fetch(`${API_BASE_URL}/shap`);
            if(!res.ok) throw new Error("SHAP data not found");
            const data = await res.json();

            const trace = {
                x: data.importance.slice(0, 10).reverse(),
                y: data.features.slice(0, 10).reverse(),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#3b82f6' }
            };
            
            const layout = {
                title: false,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#94a3b8', family: 'Inter' },
                margin: { l: 150 }
            };

            Plotly.newPlot('shap-chart', [trace], layout, {responsive: true});
        } catch(e) {
            console.warn(e);
        }
    }

    function showToast(msg) {
        const toast = document.getElementById('error-toast');
        toast.innerText = msg;
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 3500);
    }
});
