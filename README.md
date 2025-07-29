<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Chatbot Project Report</title>
  <style>
    /* General styles */
    html {
      scroll-behavior: smooth;
    }
    body {
      font-family: Arial, sans-serif;
      margin: 2em;
      background-color: #f7f7f7;
      color: #333;
      line-height: 1.6; /* Improved readability for body text */
    }

  </style>
</head>
<body>

  <section class="project-tree">
    <h2>🌲 Project Tree</h2>
    <pre class="project-tree-code"><code>.
├── .github
│   ├── CODE
│   ├── Train
│   └── Result
├── .gitignore
└── README.md

3 directories, 47 files
    </code></pre>
  </section>

  <h1 id="project-chatbot">Machine Learning and Artificial Intelligence Based Chatbot</h1>

  <div class="centered">
    <img src="assets/logo.png" alt="Chatbot Logo">
    <p>
      A multilingual AI chatbot application that intelligently handles queries in Bangla, Hindi, and English, combining multiple advanced Natural Language Processing capabilities in one platform.
    </p>
  </div>

  <section id="team-members">
    <h2>Team Members</h2>
    <h3>Group X</h3>
    <table>
      <thead>
        <tr>
          <th>Member</th>
          <th>Name</th>
          <th>ID</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>Zawed Bin Tariq</td>
          <td>2232811642</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Nazmus Shakib</td>
          <td>2312009042</td>
        </tr>
      </tbody>
    </table>
  </section>

  <section id="course-and-faculty">
    <h2>Course and Faculty</h2>
    <ul>
      <li><strong>Course:</strong> CSE299 - Junior Design</li>
      <li><strong>Semester:</strong> Summer 2025</li>
      <li><strong>Institution:</strong> North South University</li>
      <li><strong>Faculty:</strong> Mohammad Shifat-E-Rabbi</li>
    </ul>
  </section>

  <section id="table-of-contents">
    <h2>Table of Contents</h2>
    <ul>
      <li><a href="#introduction">Introduction</a></li>
      <li><a href="#system-architecture">System Architecture</a></li>
      <li><a href="#features">Features</a></li>
      <li><a href="#core-modules">Core Modules and Responsibilities</a></li> <li><a href="#installation">Installation and Running</a></li>
      <li><a href="#usage">Usage</a></li>
      <li><a href="#screenshots">Screenshots</a></li>
      <li><a href="#conclusion">Conclusion</a></li>
    </ul>
  </section>

  <section id="introduction">
    <h2>Introduction</h2>
    <p>
      In an increasingly digital world, intelligent conversational systems have become essential for enhancing user engagement and support. The AI Chatbot project was developed to demonstrate the power and flexibility of modern NLP models by offering a single chatbot capable of performing diverse tasks across multiple languages. Its primary aim is to provide meaningful, human-like interactions while also assisting with informative and functional queries such as mathematical calculations, text summarization, and data retrieval.
    </p>
  </section>

  <section id="system-architecture">
    <h2>System Architecture</h2>
    <p>
      The system follows a modular, controller-driven architecture. At the core is a central orchestrator (<code>chatbot.py</code>) that interprets user input and routes it to specialized modules. Each module handles specific types of tasks, such as language generation or summarization. Models like Hugging Face Transformers, Gemini APIs, and classical ML models are embedded seamlessly. This architecture ensures extensibility, maintainability, and separation of concerns.
    </p>
  </section>

  <section id="features">
    <h2>Features</h2>
    <ul>
      <li>Bangla and Hindi text generation using pretrained models and APIs</li>
      <li>Bangla text summarization using the T5 transformer model</li>
      <li>Drug information retrieval using semantic keyword matching in CSV datasets</li>
      <li>Rental prediction powered by a trained regression model</li>
      <li>Mathematical expression evaluation through symbolic computation</li>
      <li>Language translation between Bangla, Hindi, and English</li>
      <li>Conversational dialogue management using DialoGPT</li>
    </ul>
  </section>

  <section id="core-modules"> <h2>Core Modules and Responsibilities</h2>
    <ul>
      <li><strong><code>chatbot.py</code>:</strong> Main application file that manages command routing and input parsing.</li>
      <li><strong><code>bangla_generator.py</code>:</strong> Handles generation of text in Bangla using fine-tuned language models.</li>
      <li><strong><code>bangla_summarizer.py</code>:</strong> Summarizes long Bangla texts into concise summaries using T5.</li>
      <li><strong><code>drug_info_processor.py</code>:</strong> Enables search and retrieval from a structured drug information dataset.</li>
      <li><strong><code>hindi_generator.py</code>:</strong> Uses Gemini API to dynamically generate responses in Hindi.</li>
      <li><strong><code>math_solver.py</code>:</strong> Solves and simplifies mathematical expressions using the SymPy library.</li>
      <li><strong><code>translator.py</code>:</strong> Manages translation tasks across supported languages.</li>
      <li><strong><code>train_data_manager.py</code>:</strong> Stores historical chat data and feedback for future training iterations.</li>
    </ul>
  </section>

  <section id="installation">
    <h2>Installation and Running</h2>
    <ol>
      <li>Ensure Python 3.10+ is installed</li>
      <li>Clone the repository using <code>git clone &lt;repository_url&gt;</code></li> <li>Install dependencies with <code>pip install -r requirements.txt</code></li>
      <li>Start the chatbot with <code>python main.py</code></li>
    </ol>
  </section>

  <section id="usage">
    <h2>Usage</h2>
    <ul>
      <li>Launch the application through terminal</li>
      <li>Run a feature module via command line argument (e.g., translation, rental, generation)</li>
      <li>Sample command: <code>python main.py math "10 * (2 + 3)"</code></li>
      <li>Supports both CLI and conversational interactions</li>
    </ul>
  </section>

  <section id="screenshots">
    <h2>Screenshots</h2>
    <p><em>Include annotated screenshots of the terminal, input/output windows, and example usage interfaces.</em></p>
    <div class="centered">
        <img src="assets/screenshot1.png" alt="Screenshot 1: Terminal Interaction Example">
        <p><em>Example of a terminal interaction with the chatbot, showing input and output.</em></p>
        <img src="assets/screenshot2.png" alt="Screenshot 2: Language Translation Feature">
        <p><em>Illustrates the chatbot's language translation capability.</em></p>
        <img src="assets/screenshot3.png" alt="Screenshot 3: Math Solver in Action">
        <p><em>Demonstrates the mathematical expression evaluation feature.</em></p>
    </div>
  </section>

  <section id="conclusion">
    <h2>Conclusion</h2>
    <p>
      The AI Chatbot project is a demonstration of how diverse machine learning techniques and language models can be combined to form a multifunctional, intelligent conversational assistant. Its support for Bangla, Hindi, and English, along with its capacity to perform real-world tasks, makes it a useful academic tool and a strong candidate for practical deployment. In the future, the system can be extended with voice capabilities, deployment on web/mobile platforms, and continual model training with user feedback.
    </p>
  </section>

</body>
</html>
