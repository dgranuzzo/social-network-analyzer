
# WappAnalyzer

Python code to analyze whatsapp exported files from group conversations.

This repository contains two main modules:

- `bullying_detector.py`: Script for detecting bullying in texts using natural language processing techniques.
- `wapp_analyzer.py`: Tool for analyzing WhatsApp data, focused on identifying patterns and behaviors in conversations.

## Important Notice

This tool is designed for research and legitimate analysis purposes only. By using this software, you agree to:

- Use it in an ethical and responsible manner
- Respect user privacy and data protection regulations
- Comply with WhatsApp's terms of service
- Not use the tool for malicious purposes or harassment

## Features

- Detection of potentially toxic interactions
- Analysis of conversation patterns
- Customizable lexicons and detection patterns
- Time-based analysis of message patterns
- User interaction analysis
- Word cloud generation
- Response time analysis

## Installation

1. Clone the repository:
   ```powershell
   git clone <repository-url>
   ```
2. Install the dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

- Run the bullying detector:
  ```powershell
  python bullying_detector.py
  ```
- Run the WhatsApp analyzer:
  ```powershell
  python wapp_analyzer.py path/to/chat.txt --tz America/Sao_Paulo
  ```

### Customization

The tool includes default lexicons and patterns that can be customized:

1. Toxic Language Detection:
   - Edit `BAD_WORDS` in `bullying_detector.py` to customize the toxic words dictionary
   - Modify `BAD_PATTERNS` to adjust toxic phrase patterns
   - Adjust `toxic_threshold` parameter when calling `detect_bullying()` for sensitivity

2. Message Analysis:
   - Customize `PROFANITY` set in `wapp_analyzer.py` for profanity detection
   - Modify `MEDIA_PATTERNS` for different media message formats
   - Adjust time windows and thresholds in analysis functions

## Structure

```
bullying_detector.py        # Bullying detector
wapp_analyzer.py           # WhatsApp conversation analyzer
requirements.txt           # Project dependencies
__pycache__/              # Python cache files
```

## Output

The analyzer generates various outputs in an 'out' directory:
- CSV files with detailed analysis
- Word clouds for users and overall chat
- Activity heatmaps
- User interaction graphs
- Suspicious behavior reports

## Privacy Considerations

- All analysis is performed locally on your machine
- No data is sent to external servers
- Always obtain necessary permissions before analyzing group conversations
- Remove or anonymize sensitive information before sharing results
- Consider data protection regulations in your jurisdiction

## Contribution

Feel free to open issues or submit pull requests with improvements, bug fixes, or new features. Please ensure any contributions maintain the ethical guidelines of the project.

## License

This project is licensed under the MIT License. However, usage must comply with the responsible use guidelines outlined above.