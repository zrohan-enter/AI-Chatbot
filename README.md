<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Chatbot Project Report</title>
  <style>
    html { scroll-behavior: smooth; }
    body {
      font-family: Arial, sans-serif;
      margin: 2em;
      background-color: #f7f7f7;
      color: #333;
    }
    h1, h2, h3 { color: #2c3e50; }
    ul, ol { margin-bottom: 1em; }
    table, th, td {
      border: 1px solid #ccc;
      border-collapse: collapse;
      padding: 10px;
    }
    table { margin-bottom: 20px; width: 100%; }
    a { color: #2980b9; text-decoration: none; }
    a:hover { text-decoration: underline; }
    code, pre {
      background: #eee;
      padding: 5px;
      display: block;
      overflow-x: auto;
    }
    section { margin-bottom: 2em; }
    .centered {
      text-align: center;
    }
    .centered img {
      max-width: 250px;
      margin: 10px auto;
      display: block;
    }
  </style>
</head>
  ### 🌲 **Project tree**


```text
.
├── .github
│   ├── CODE
│   ├── Train
│   └── Result
├── .gitignore
└── README.md

3 directories, 47 files
```

---

<body>
  <h1 id="project-chatbot">Machine Learning and Artificial Intelligence Based Chatbot</h1>

  <div class="centered">
    <img src="assets/logo.png" alt="Chatbot Logo">
    <p>
      A multilingual AI chatbot application that intelligently handles queries in Bangla, Hindi, and English, combining multiple advanced Natural Language Processing capabilities in one platform.
    </p>
  </div>

  <h2 id="team-members">Team Members</h2>
  <h3>Group X</h3>
  <table>
    <tr><th>Member</th><th>Name</th><th>ID</th></tr>
    <tr><td>1</td><td>Zawed Bin Tariq</td><td>2232811642</td></tr>
    <tr><td>2</td><td>Nazmus Shakib</td><td>2312009042</td></tr>
  </table>

  <h2 id="course-and-faculty">Course and Faculty</h2>
  <ul>
    <li><strong>Course:</strong> CSE299 - Junior Design</li>
    <li><strong>Semester:</strong> Summer 2025</li>
    <li><strong>Institution:</strong> North South University</li>
    <li><strong>Faculty:</strong> Mohammad Shifat-E-Rabbi</li>
  </ul>

  <h2 id="table-of-contents">Table of Contents</h2>
  <ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#system-architecture">System Architecture</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#classes">Core Modules and Responsibilities</a></li>
    <li><a href="#installation">Installation and Running</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#screenshots">Screenshots</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
  </ul>

  <h2 id="introduction">Introduction</h2>
  <p>
    In an increasingly digital world, intelligent conversational systems have become essential for enhancing user engagement and support. The AI Chatbot project was developed to demonstrate the power and flexibility of modern NLP models by offering a single chatbot capable of performing diverse tasks across multiple languages. Its primary aim is to provide meaningful, human-like interactions while also assisting with informative and functional queries such as mathematical calculations, text summarization, and data retrieval.
  </p>

  <h2 id="system-architecture">System Architecture</h2>
  <p>
    The system follows a modular, controller-driven architecture. At the core is a central orchestrator (<code>chatbot.py</code>) that interprets user input and routes it to specialized modules. Each module handles specific types of tasks, such as language generation or summarization. Models like Hugging Face Transformers, Gemini APIs, and classical ML models are embedded seamlessly. This architecture ensures extensibility, maintainability, and separation of concerns.
  </p>

  <h2 id="features">Features</h2>
  <ul>
    <li>Bangla and Hindi text generation using pretrained models and APIs</li>
    <li>Bangla text summarization using the T5 transformer model</li>
    <li>Drug information retrieval using semantic keyword matching in CSV datasets</li>
    <li>Rental prediction powered by a trained regression model</li>
    <li>Mathematical expression evaluation through symbolic computation</li>
    <li>Language translation between Bangla, Hindi, and English</li>
    <li>Conversational dialogue management using DialoGPT</li>
  </ul>

  <h2 id="classes">Core Modules and Responsibilities</h2>
  <ul>
    <li><strong>chatbot.py:</strong> Main application file that manages command routing and input parsing.</li>
    <li><strong>bangla_generator.py:</strong> Handles generation of text in Bangla using fine-tuned language models.</li>
    <li><strong>bangla_summarizer.py:</strong> Summarizes long Bangla texts into concise summaries using T5.</li>
    <li><strong>drug_info_processor.py:</strong> Enables search and retrieval from a structured drug information dataset.</li>
    <li><strong>hindi_generator.py:</strong> Uses Gemini API to dynamically generate responses in Hindi.</li>
    <li><strong>math_solver.py:</strong> Solves and simplifies mathematical expressions using the SymPy library.</li>
    <li><strong>translator.py:</strong> Manages translation tasks across supported languages.</li>
    <li><strong>train_data_manager.py:</strong> Stores historical chat data and feedback for future training iterations.</li>
  </ul>

  <h2 id="installation">Installation and Running</h2>
  <ol>
    <li>Ensure Python 3.10+ is installed</li>
    <li>Clone the repository using <code>git clone</code></li>
    <li>Install dependencies with <code>pip install -r requirements.txt</code></li>
    <li>Start the chatbot with <code>python main.py</code></li>
  </ol>

  <h2 id="usage">Usage</h2>
  <ul>
    <li>Launch the application through terminal</li>
    <li>Run a feature module via command line argument (e.g., translation, rental, generation)</li>
    <li>Sample command: <code>python main.py math "10 * (2 + 3)"</code></li>
    <li>Supports both CLI and conversational interactions</li>
  </ul>

  <h2 id="screenshots">Screenshots</h2>
  <p><em>Include annotated screenshots of the terminal, input/output windows, and example usage interfaces.</em></p>

  <h2 id="conclusion">Conclusion</h2>
  <p>
    The AI Chatbot project is a demonstration of how diverse machine learning techniques and language models can be combined to form a multifunctional, intelligent conversational assistant. Its support for Bangla, Hindi, and English, along with its capacity to perform real-world tasks, makes it a useful academic tool and a strong candidate for practical deployment. In the future, the system can be extended with voice capabilities, deployment on web/mobile platforms, and continual model training with user feedback.
  </p>
</body>
</html>




