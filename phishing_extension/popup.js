document.getElementById('analyzeButton').addEventListener('click', async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: extractEmailContent
    }, async (injectionResults) => {
        if (injectionResults && injectionResults[0] && injectionResults[0].result) {
            const emailData = injectionResults[0].result;

            // not on email page case
            if (emailData.error) {
                document.getElementById('result').innerText = emailData.error;
                return;
            }

            try {
                
                const response = await fetch('http://127.0.0.1:8000/predict', { 
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(emailData)
                });

                if (response.ok) {
                    const result = await response.json();
                    document.getElementById('result').innerText = `Prediction: ${result.is_phishing ? 'Phishing' : 'Safe'}\nProbability: ${result.phishing_probability.toFixed(2)}\nConfidence: ${result.confidence}`;
                } else {
                    document.getElementById('result').innerText = 'Error contacting detection server.';
                }
            } catch (error) {
                document.getElementById('result').innerText = 'Error connecting to the backend server.';
                console.error(error);
            }
        } else {
            document.getElementById('result').innerText = 'Failed to extract email content.';
        }
    });
});

function extractEmailContent() {
    try {
        
        let subjectElement = document.querySelector('h2.hP');
        let bodyElement = document.querySelector('div.a3s');

        // error if no gmail elements are found
        if (!subjectElement || !bodyElement) {
            return { error: "Please open an email first before using the Phishing Detector." };
        }

        // Extract sender, subject, and body
        let sender = document.querySelector('span[email]')?.getAttribute('email') || '';
        let subject = subjectElement.innerText || '';
        let body = bodyElement.innerText || '';

        return { sender, subject, body };
    } catch (error) {
        console.error("Error extracting email content:", error);
        return { error: "Something went wrong while reading the email content." };
    }
}
