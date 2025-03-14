from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import logging
import json
from enum import Enum
import google.generativeai as genai
from datetime import datetime, timedelta
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')  # For Claude
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # For Google Cloud Translation
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Separate key for Gemini
logger.info(f"Loaded API Keys - OpenAI: {'Present' if OPENAI_API_KEY else 'Missing'}, "
           f"Anthropic: {'Present' if ANTHROPIC_API_KEY else 'Missing'}, "
           f"Google Cloud: {'Present' if GOOGLE_API_KEY else 'Missing'}, "
           f"Gemini: {'Present' if GEMINI_API_KEY else 'Missing'}")

# Setup API
app = FastAPI()

class ModelType(str, Enum):
    GPT35 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    CLAUDE = "claude-3-opus-20240229"
    GEMINI = "gemini-1.5-pro"
    GOOGLE = "google-translate"

# Setup templates
templates = Jinja2Templates(directory="templates")

# Statistics tracking
class Stats:
    def __init__(self):
        self.stats_file = "stats_data.json"  # This is our memory file
        self.load_stats()  # Load existing stats when server starts

    def load_stats(self):
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.total_translations = data.get('total_translations', 0)
                    self.total_edits = data.get('total_edits', 0)
                    self.active_users = set(data.get('active_users', []))
                    self.model_usage = defaultdict(int, data.get('model_usage', {}))
                    self.recent_activities = data.get('recent_activities', [])
                    self.successful_requests = data.get('successful_requests', 0)
                    self.total_requests = data.get('total_requests', 0)
                    self.model_success_rates = defaultdict(lambda: {"success": 0, "total": 0}, 
                                                        data.get('model_success_rates', {}))
                    self.model_response_times = defaultdict(list, 
                                                         {k: v for k, v in data.get('model_response_times', {}).items()})
                    self.model_quality_scores = defaultdict(list, 
                                                         {k: v for k, v in data.get('model_quality_scores', {}).items()})
                    self.model_error_types = defaultdict(lambda: defaultdict(int), 
                                                       data.get('model_error_types', {}))
                    self.mode_metrics = data.get('mode_metrics', {
                        "fast_mode": {
                            "success_rate": 0,
                            "total_requests": 0,
                            "successful_requests": 0,
                            "response_times": [],
                            "quality_scores": []
                        },
                        "detailed_mode": {
                            "success_rate": 0,
                            "total_requests": 0,
                            "successful_requests": 0,
                            "response_times": [],
                            "quality_scores": []
                        }
                    })
                    # Add time-based tracking with proper defaultdict initialization
                    self.daily_stats = {
                        date: {
                            **stats,
                            'model_usage': defaultdict(int, stats.get('model_usage', {})),
                            'quality_scores': defaultdict(list, stats.get('quality_scores', {}))
                        }
                        for date, stats in data.get('daily_stats', {}).items()
                    }
                    self.weekly_stats = {
                        week: {
                            **stats,
                            'model_usage': defaultdict(int, stats.get('model_usage', {})),
                            'quality_scores': defaultdict(list, stats.get('quality_scores', {}))
                        }
                        for week, stats in data.get('weekly_stats', {}).items()
                    }
            else:
                # Initialize with default values if no file exists
                self.total_translations = 0
                self.total_edits = 0
                self.active_users = set()
                self.model_usage = defaultdict(int)
                self.recent_activities = []
                self.successful_requests = 0
                self.total_requests = 0
                self.model_success_rates = defaultdict(lambda: {"success": 0, "total": 0})
                self.model_response_times = defaultdict(list)
                self.model_quality_scores = defaultdict(list)
                self.model_error_types = defaultdict(lambda: defaultdict(int))
                self.mode_metrics = {
                    "fast_mode": {
                        "success_rate": 0,
                        "total_requests": 0,
                        "successful_requests": 0,
                        "response_times": [],
                        "quality_scores": []
                    },
                    "detailed_mode": {
                        "success_rate": 0,
                        "total_requests": 0,
                        "successful_requests": 0,
                        "response_times": [],
                        "quality_scores": []
                    }
                }
                # Initialize time-based tracking with proper defaultdict initialization
                self.daily_stats = {}
                self.weekly_stats = {}
        except Exception as e:
            logger.error(f"Error loading stats: {str(e)}")
            # Initialize with default values if loading fails
            self.total_translations = 0
            self.total_edits = 0
            self.active_users = set()
            self.model_usage = defaultdict(int)
            self.recent_activities = []
            self.successful_requests = 0
            self.total_requests = 0
            self.model_success_rates = defaultdict(lambda: {"success": 0, "total": 0})
            self.model_response_times = defaultdict(list)
            self.model_quality_scores = defaultdict(list)
            self.model_error_types = defaultdict(lambda: defaultdict(int))
            self.mode_metrics = {
                "fast_mode": {
                    "success_rate": 0,
                    "total_requests": 0,
                    "successful_requests": 0,
                    "response_times": [],
                    "quality_scores": []
                },
                "detailed_mode": {
                    "success_rate": 0,
                    "total_requests": 0,
                    "successful_requests": 0,
                    "response_times": [],
                    "quality_scores": []
                }
            }
            # Initialize time-based tracking with proper defaultdict initialization
            self.daily_stats = {}
            self.weekly_stats = {}

    def save_stats(self):
        # Save all stats to JSON file
        data = {
            'total_translations': self.total_translations,
            'total_edits': self.total_edits,
            'active_users': list(self.active_users),
            'model_usage': dict(self.model_usage),
            'recent_activities': self.recent_activities,
            'successful_requests': self.successful_requests,
            'total_requests': self.total_requests,
            'model_success_rates': dict(self.model_success_rates),
            'model_response_times': dict(self.model_response_times),
            'model_quality_scores': dict(self.model_quality_scores),
            'model_error_types': dict(self.model_error_types),
            'mode_metrics': self.mode_metrics,
            'daily_stats': {
                date: {
                    **stats,
                    'model_usage': dict(stats['model_usage']),
                    'quality_scores': dict(stats['quality_scores'])
                }
                for date, stats in self.daily_stats.items()
            },
            'weekly_stats': {
                week: {
                    **stats,
                    'model_usage': dict(stats['model_usage']),
                    'quality_scores': dict(stats['quality_scores'])
                }
                for week, stats in self.weekly_stats.items()
            }
        }
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update_time_based_stats(self, activity_type: str, model: str, success: bool, quality_score: int = None):
        current_date = datetime.now()
        date_str = current_date.strftime("%Y-%m-%d")
        week_str = current_date.strftime("%Y-W%W")  # Format: YYYY-WNN (week number)

        # Convert model to string if it's a ModelType enum
        model_str = model.value if hasattr(model, 'value') else str(model)

        # Update daily stats
        if date_str not in self.daily_stats:
            self.daily_stats[date_str] = {
                "translations": 0,
                "edits": 0,
                "successful_requests": 0,
                "total_requests": 0,
                "model_usage": defaultdict(int),
                "quality_scores": defaultdict(list)
            }
        
        daily = self.daily_stats[date_str]
        if activity_type == "Translation":
            daily["translations"] += 1
        else:
            daily["edits"] += 1
        daily["model_usage"][model_str] += 1
        daily["total_requests"] += 1
        if success:
            daily["successful_requests"] += 1
        if quality_score:
            daily["quality_scores"][model_str].append(quality_score)

        # Update weekly stats
        if week_str not in self.weekly_stats:
            self.weekly_stats[week_str] = {
                "translations": 0,
                "edits": 0,
                "successful_requests": 0,
                "total_requests": 0,
                "model_usage": defaultdict(int),
                "quality_scores": defaultdict(list)
            }
        
        weekly = self.weekly_stats[week_str]
        if activity_type == "Translation":
            weekly["translations"] += 1
        else:
            weekly["edits"] += 1
        weekly["model_usage"][model_str] += 1
        weekly["total_requests"] += 1
        if success:
            weekly["successful_requests"] += 1
        if quality_score:
            weekly["quality_scores"][model_str].append(quality_score)

        # Clean up old stats (keep last 30 days and 12 weeks)
        self._cleanup_old_stats()

    def _cleanup_old_stats(self):
        current_date = datetime.now()
        # Remove daily stats older than 30 days
        self.daily_stats = {k: v for k, v in self.daily_stats.items() 
                          if (current_date - datetime.strptime(k, "%Y-%m-%d")).days <= 30}
        # Remove weekly stats older than 12 weeks
        self.weekly_stats = {k: v for k, v in self.weekly_stats.items() 
                           if (current_date - datetime.strptime(k, "%Y-W%W")).days <= 84}

    def get_time_based_metrics(self):
        try:
            current_date = datetime.now()
            date_str = current_date.strftime("%Y-%m-%d")
            week_str = current_date.strftime("%Y-W%W")

            # Get today's stats with default values
            today_stats = self.daily_stats.get(date_str, {
                "translations": 0,
                "edits": 0,
                "successful_requests": 0,
                "total_requests": 0,
                "model_usage": defaultdict(int),
                "quality_scores": defaultdict(list)
            })

            # Get this week's stats with default values
            week_stats = self.weekly_stats.get(week_str, {
                "translations": 0,
                "edits": 0,
                "successful_requests": 0,
                "total_requests": 0,
                "model_usage": defaultdict(int),
                "quality_scores": defaultdict(list)
            })

            # Calculate averages with error handling
            def calculate_metrics(stats):
                try:
                    total_requests = stats["total_requests"]
                    successful_requests = stats["successful_requests"]
                    success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
                    
                    # Calculate average quality scores per model
                    avg_quality_scores = {}
                    for model, scores in stats["quality_scores"].items():
                        if scores:
                            avg_quality_scores[model] = sum(scores) / len(scores)
                        else:
                            avg_quality_scores[model] = 0

                    return {
                        "translations": stats["translations"],
                        "edits": stats["edits"],
                        "success_rate": round(success_rate, 1),
                        "model_usage": dict(stats["model_usage"]),
                        "avg_quality_scores": avg_quality_scores
                    }
                except Exception as e:
                    logger.error(f"Error calculating metrics: {str(e)}")
                    return {
                        "translations": 0,
                        "edits": 0,
                        "success_rate": 0,
                        "model_usage": {},
                        "avg_quality_scores": {}
                    }

            return {
                "today": calculate_metrics(today_stats),
                "this_week": calculate_metrics(week_stats),
                "total": {
                    "translations": self.total_translations,
                    "edits": self.total_edits,
                    "success_rate": round(self.get_api_success_rate(), 1),
                    "model_usage": dict(self.model_usage),
                    "avg_quality_scores": {
                        model: sum(scores) / len(scores) if scores else 0
                        for model, scores in self.model_quality_scores.items()
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error getting time-based metrics: {str(e)}")
            return {
                "today": {
                    "translations": 0,
                    "edits": 0,
                    "success_rate": 0,
                    "model_usage": {},
                    "avg_quality_scores": {}
                },
                "this_week": {
                    "translations": 0,
                    "edits": 0,
                    "success_rate": 0,
                    "model_usage": {},
                    "avg_quality_scores": {}
                },
                "total": {
                    "translations": 0,
                    "edits": 0,
                    "success_rate": 0,
                    "model_usage": {},
                    "avg_quality_scores": {}
                }
            }

    def add_translation(self, model: str, success: bool = True, response_time: float = 0, error_type: str = None, quality_score: int = None):
        self.total_translations += 1
        
        # Convert model to string if it's a ModelType enum
        model_str = model.value if hasattr(model, 'value') else str(model)
        
        self.model_usage[model_str] += 1
        self.model_success_rates[model_str]["total"] += 1
        if success:
            self.model_success_rates[model_str]["success"] += 1
        if response_time > 0:
            self.model_response_times[model_str].append(response_time)
        if error_type:
            if model_str not in self.model_error_types:
                self.model_error_types[model_str] = defaultdict(int)
            self.model_error_types[model_str][error_type] += 1
        if quality_score:
            self.model_quality_scores[model_str].append(quality_score)
            
        self.add_activity("Translation", f"Translation using {model_str}")
        self.update_time_based_stats("Translation", model_str, success, quality_score)
        self.save_stats()

    def add_edit(self, model: str, mode: str, success: bool = True, response_time: float = 0, error_type: str = None, quality_score: int = None):
        self.total_edits += 1
        
        # Convert model to string if it's a ModelType enum
        model_str = model.value if hasattr(model, 'value') else str(model)
        
        self.model_usage[model_str] += 1
        self.model_success_rates[model_str]["total"] += 1
        
        # Update model-specific metrics
        if success:
            self.model_success_rates[model_str]["success"] += 1
        if response_time > 0:
            self.model_response_times[model_str].append(response_time)
        if error_type:
            if model_str not in self.model_error_types:
                self.model_error_types[model_str] = defaultdict(int)
            self.model_error_types[model_str][error_type] += 1
        if quality_score:
            self.model_quality_scores[model_str].append(quality_score)
            
        # Update mode-specific metrics
        mode_key = "fast_mode" if mode == "fast" else "detailed_mode"
        self.mode_metrics[mode_key]["total_requests"] += 1
        if success:
            self.mode_metrics[mode_key]["successful_requests"] += 1
        if response_time > 0:
            self.mode_metrics[mode_key]["response_times"].append(response_time)
        if quality_score:
            self.mode_metrics[mode_key]["quality_scores"].append(quality_score)
            
        self.add_activity("Edit", f"Text edit using {model_str} in {mode} mode")
        self.update_time_based_stats("Edit", model_str, success, quality_score)
        self.save_stats()

    def add_user(self, user_id: str):
        self.active_users.add(user_id)
        self.save_stats()  # Save after updating

    def add_activity(self, activity_type: str, description: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.recent_activities.insert(0, {
            "type": activity_type,
            "description": description,
            "timestamp": timestamp
        })
        # Keep only last 10 activities
        self.recent_activities = self.recent_activities[:10]
        self.save_stats()  # Save after updating

    def record_request(self, success: bool):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        self.save_stats()  # Save after updating

    def get_api_success_rate(self) -> float:
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    def add_quality_score(self, model: str, score: int):
        if 1 <= score <= 5:
            self.model_quality_scores[model].append(score)

    def get_model_metrics(self):
        metrics = {}
        
        # Add mode-specific metrics
        for mode, data in self.mode_metrics.items():
            success_rate = (data["successful_requests"] / data["total_requests"] * 100) if data["total_requests"] > 0 else 0
            avg_response_time = sum(data["response_times"]) / len(data["response_times"]) if data["response_times"] else 0
            avg_quality = sum(data["quality_scores"]) / len(data["quality_scores"]) if data["quality_scores"] else 0
            
            metrics[mode] = {
                "success_rate": round(success_rate, 1),
                "avg_response_time": round(avg_response_time, 2),
                "avg_quality": round(avg_quality, 1)
            }
        
        # Add model-specific metrics
        for model in self.model_usage.keys():
            success_rate = (self.model_success_rates[model]["success"] / 
                          self.model_success_rates[model]["total"] * 100) if self.model_success_rates[model]["total"] > 0 else 0
            avg_response_time = sum(self.model_response_times[model]) / len(self.model_response_times[model]) if self.model_response_times[model] else 0
            avg_quality = sum(self.model_quality_scores[model]) / len(self.model_quality_scores[model]) if self.model_quality_scores[model] else 0
            
            metrics[model] = {
                "success_rate": round(success_rate, 1),
                "avg_response_time": round(avg_response_time, 2),
                "avg_quality": round(avg_quality, 1),
                "total_requests": self.model_success_rates[model]["total"],
                "error_types": dict(self.model_error_types[model])
            }
        
        return metrics

stats = Stats()

# Create templates directory and HTML file
os.makedirs("templates", exist_ok=True)
with open("templates/index.html", "w") as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Persian Text Editor and Translator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            background-color: #f5f5f5;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .text-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .text-section:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        textarea {
            width: 100%;
            height: 120px;
            margin: 10px 0;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            resize: vertical;
            transition: border-color 0.3s ease;
        }
        textarea:focus {
            border-color: #2196F3;
            outline: none;
        }
        .edited-text {
            width: 100%;
            min-height: 50px;
            margin: 15px 0;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            background-color: #f8f9fa;
            text-align: right;
            direction: rtl;
        }
        .explanation {
            margin-top: 15px;
            padding: 12px;
            background-color: #e3f2fd;
            border-radius: 8px;
            font-size: 14px;
            color: #1976D2;
            text-align: left;
            direction: ltr;
            border-left: 4px solid #1976D2;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 8px;
        }
        button:hover {
            background-color: #1976D2;
            transform: translateY(-1px);
        }
        button:active {
            transform: translateY(1px);
        }
        .loading {
            color: #666;
            font-style: italic;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .loading:after {
            content: "...";
            animation: dots 1.5s steps(5, end) infinite;
        }
        @keyframes dots {
            0%, 20% { content: "."; }
            40% { content: ".."; }
            60%, 100% { content: "..."; }
        }
        .error {
            color: #d32f2f;
            font-weight: bold;
            padding: 8px;
            border-radius: 4px;
            background-color: #ffebee;
            border: 1px solid #ffcdd2;
        }
        select {
            padding: 10px;
            font-size: 16px;
            margin: 10px 0;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            background-color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        select:hover {
            border-color: #2196F3;
        }
        .mode-selection, .model-selection {
            margin: 15px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .mode-info, .model-info {
            margin-top: 8px;
            font-size: 14px;
            color: #666;
            line-height: 1.4;
        }
        .section-title {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        .mode-selection {
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .mode-info {
            margin-top: 5px;
            font-size: 14px;
            color: #666;
        }
        #editMode {
            padding: 8px;
            font-size: 16px;
            border-radius: 5px;
            border: 2px solid #ddd;
            width: 100%;
            max-width: 300px;
        }
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-top: 20px;
        }
        .result-box {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
        }
        .result-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .edited-text {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 0;
            font-size: 16px;
            line-height: 1.6;
        }
        .explanation {
            background-color: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 8px;
            padding: 15px;
            margin: 0;
            font-size: 14px;
            line-height: 1.6;
            color: #1976D2;
        }
    </style>
</head>
<body>
    <h1>Persian Text Editor and Translator</h1>
    <div class="container">
        <!-- Persian Text Editor Section -->
        <div class="text-section">
            <div class="section-title">Persian Text Editor</div>
            <div class="mode-selection">
                <select id="editMode">
                    <option value="fast">Fast Edit (Grammar & Spelling)</option>
                    <option value="detailed">Detailed Edit (Professional & Coaching)</option>
                </select>
                <div class="mode-info">Fast mode: Quick grammar and spelling fixes. Detailed mode: Deep professional content enhancement.</div>
            </div>
            <div class="model-selection">
                <select id="editModel">
                    <option value="gemini-1.5-pro" selected>Gemini Pro (Advanced AI)</option>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Fast & Good)</option>
                    <option value="gpt-4">GPT-4 (Most Accurate)</option>
                    <option value="claude-3-opus-20240229">Claude-3 (Accurate)</option>
                </select>
                <div id="editModelInfo" class="model-info"></div>
            </div>
            <textarea id="persianText" placeholder="Write your Persian text here..." dir="rtl"></textarea>
            <button id="editButton" class="edit">Edit Text</button>
            
            <div class="results-container">
                <div class="result-box">
                    <div class="result-title">Edited Text:</div>
                    <div id="editedText" class="edited-text" dir="rtl"></div>
                </div>
                <div class="result-box">
                    <div class="result-title">Explanation:</div>
                    <div id="explanation" class="explanation"></div>
                </div>
            </div>
        </div>

        <!-- Translation Section -->
        <div class="text-section">
            <div class="section-title">Translate to English</div>
            <div class="model-selection">
                <select id="model">
                    <option value="gemini-1.5-pro" selected>Gemini Pro (Advanced AI)</option>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Fast & Good)</option>
                    <option value="gpt-4">GPT-4 (Most Accurate)</option>
                    <option value="claude-3-opus-20240229">Claude-3 (Accurate)</option>
                    <option value="google-translate">Google Cloud Translation (Basic)</option>
                </select>
                <div id="modelInfo" class="model-info"></div>
            </div>
            <button id="translateButton">Translate</button>
            <div id="result"></div>
        </div>
    </div>

    <script>
        // Wait for DOM to load
        window.addEventListener("DOMContentLoaded", () => {
            // Initialize model info
            updateModelInfo();

            // Add event listeners
            document.getElementById("editButton").addEventListener("click", editText);
            document.getElementById("translateButton").addEventListener("click", translateText);
            document.getElementById("model").addEventListener("change", updateModelInfo);
            document.getElementById("editModel").addEventListener("change", updateModelInfo);
            document.getElementById("persianText").addEventListener("keypress", (e) => {
                if (e.key === "Enter" && e.ctrlKey) {
                    e.preventDefault();
                    editText();
                }
            });
        });

        // Update model information display
        function updateModelInfo() {
            const model = document.getElementById("model").value;
            const modelInfo = document.getElementById("modelInfo");
            const editModel = document.getElementById("editModel").value;
            const editModelInfo = document.getElementById("editModelInfo");
            
            const modelDescriptions = {
                "gpt-3.5-turbo": "Fast and reliable processing with good accuracy",
                "gpt-4": "Most accurate processing, better understanding of context and nuances",
                "claude-3-opus-20240229": "Advanced AI model with strong understanding of Persian language",
                "google-translate": "Basic machine translation, good for simple texts",
                "gemini-1.5-pro": "Google's advanced AI model, excellent for context and cultural nuances"
            };
            
            modelInfo.textContent = modelDescriptions[model];
            editModelInfo.textContent = modelDescriptions[editModel];
        }

        // Handle text editing
        async function editText() {
            const text = document.getElementById("persianText").value;
            const editMode = document.getElementById("editMode").value;
            const editModel = document.getElementById("editModel").value;
            const editedText = document.getElementById("editedText");
            const explanation = document.getElementById("explanation");
            
            if (!text.trim()) {
                editedText.innerHTML = '<span class="error">لطفا متن فارسی را وارد کنید</span>';
                explanation.innerHTML = "";
                return;
            }
            
            editedText.innerHTML = '<span class="loading">در حال ویرایش متن...</span>';
            explanation.innerHTML = "";
            
            try {
                const response = await fetch("/edit", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ 
                        text: text,
                        mode: editMode,
                        model: editModel
                    })
                });
                
                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.detail || "Edit failed");
                }
                
                const data = await response.json();
                editedText.textContent = data.edited_text;
                explanation.textContent = data.explanation;
            } catch (error) {
                editedText.innerHTML = `<span class="error">خطا: ${error.message}</span>`;
                explanation.innerHTML = "";
                console.error("Edit error:", error);
            }
        }

        // Handle text translation
        async function translateText() {
            const text = document.getElementById("editedText").textContent || document.getElementById("persianText").value;
            const model = document.getElementById("model").value;
            const result = document.getElementById("result");
            
            if (!text.trim()) {
                result.innerHTML = '<span class="error">Please enter some text to translate</span>';
                return;
            }
            
            result.innerHTML = '<span class="loading">Translating...</span>';
            
            try {
                const response = await fetch("/translate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        text: text,
                        model: model
                    })
                });
                
                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.detail || "Translation failed");
                }
                
                const data = await response.json();
                result.innerHTML = data.translated_text;
            } catch (error) {
                result.innerHTML = `<span class="error">❌ Error: ${error.message}</span>`;
                console.error("Translation error:", error);
            }
        }
    </script>
</body>
</html>
''')

class EditRequest(BaseModel):
    text: str
    mode: str  # "fast" or "detailed"
    model: str  # Changed from ModelType to str to accept the raw model name

class TranslationRequest(BaseModel):
    text: str
    model: str  # Changed from ModelType to str to accept the raw model name

def get_system_prompt(mode: str, language: str) -> str:
    return f"""You are an expert bilingual editor specializing in {language} professional development, coaching, and psychological content.

EXPERTISE AREAS:
1. Professional Terminology:
   - Coaching and psychological terms
   - Leadership development vocabulary
   - Self-help and motivational language
   - Technical accuracy in both languages

2. Content Style:
   - Professional yet engaging tone
   - Clear and authoritative voice
   - Appropriate formality level
   - Cultural sensitivity

3. Genre-Specific Knowledge:
   - Self-help book standards
   - Coaching methodology
   - Psychological concepts
   - Motivational writing

MODE SPECIFICATIONS:
For '{mode}' mode in {language}:
- Fast Mode: 
  • Basic grammar and spelling fixes
  • Simple terminology corrections
  • Fix punctuation and spacing
  • Quick surface improvements
  • Minimal technical adjustments

- Detailed Mode:
  • Precise professional terminology refinement
  • Careful tone adjustment for coaching context
  • Enhanced clarity without restructuring
  • Professional language improvement
  • Coaching terminology accuracy
  • Cultural nuance preservation
  • Technical term precision
  • Maintain original message and structure
  • Polish existing expressions
  • Subtle improvements to flow

LANGUAGE-SPECIFIC GUIDELINES:
For Persian (فارسی):
- حفظ لحن رسمی و حرفه‌ای
- استفاده صحیح از اصطلاحات تخصصی کوچینگ
- رعایت ساختار نگارشی متون روانشناسی
- حفظ انسجام در ترجمه مفاهیم تخصصی

For English:
- Maintain professional coaching terminology
- Ensure psychological concept accuracy
- Preserve motivational impact
- Balance technical and accessible language"""

def get_edit_prompt(text: str, mode: str, language: str) -> str:
    return f"""Edit this {language} professional development text in {mode} mode.

EDITING GUIDELINES:
1. Maintain subject matter expertise in:
   - Coaching methodology
   - Psychological concepts
   - Professional development
   - Leadership principles

2. Ensure appropriate:
   - Technical terminology
   - Professional tone
   - Concept clarity
   - Engagement level

3. Preserve:
   - Core message integrity
   - Professional credibility
   - Motivational impact
   - Cultural context

Text to edit:
{text}

Return only the edited text without explanations."""

def get_explanation_prompt(original_text: str, edited_text: str, language: str) -> str:
    return f"""Provide a brief, bullet-point summary of key changes made to this {language} text:

• Grammar & Style:
  - List 1-2 major grammar/style improvements

• Terminology:
  - Note any professional term improvements

• Tone & Impact:
  - Mention key tone/impact enhancements

Keep the explanation short and focused on the most important changes.

Original Text:
{original_text}

Edited Text:
{edited_text}

Provide a concise bullet-point summary."""

async def translate_with_openai(text: str, model: str) -> str:
    try:
        if not OPENAI_API_KEY:
            raise Exception("OpenAI API key is missing")
            
        temperature = 0.1  # Very low temperature for exact translations

        system_prompt = """You are an expert translator specializing in professional development, coaching, and psychological content. 

IMPORTANT: This is a direct translation task. Focus on:
- Exact meaning preservation
- Precise terminology mapping
- Professional coaching and psychological terms
- No creative variations or generations
- Maintain source text structure when possible"""

        user_prompt = f"""Translate this text while maintaining exact meaning and terminology:
1. Use precise professional/coaching terms
2. Maintain source text structure
3. Preserve exact concepts
4. Keep cultural context
5. No creative additions

Text to translate:
{text}

Provide only the direct translation."""

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        logger.error(f"OpenAI translation error: {str(e)}")
        raise Exception(f"OpenAI translation failed: {str(e)}")

async def translate_with_claude(text: str) -> str:
    try:
        if not ANTHROPIC_API_KEY:
            raise Exception("Anthropic API key is missing")
            
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": ANTHROPIC_API_KEY
        }
        
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "temperature": 0.1,  # Very low for exact translations
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise translator focusing on exact terminology and accurate translations. Maintain maximum accuracy and conciseness."
                },
                {
                    "role": "user",
                    "content": f"Translate this Persian text to English with precise terminology and maximum accuracy: {text}"
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result["content"][0]["text"].strip()
        
    except Exception as e:
        logger.error(f"Claude translation error: {str(e)}")
        raise Exception(f"Claude translation failed: {str(e)}")

async def translate_with_google_cloud(text: str) -> str:  # Renamed from translate_with_gemini
    try:
        logger.info("Starting Google Cloud Translation")
        url = "https://translation.googleapis.com/language/translate/v2"
        params = {
            'q': text,
            'target': 'en',
            'source': 'fa',
            'key': GOOGLE_API_KEY
        }
        response = requests.post(url, params=params)
        response.raise_for_status()
        result = response.json()
        translation = result['data']['translations'][0]['translatedText'].strip()
        logger.info(f"Google Cloud Translation successful: {translation}")
        return translation
    except Exception as e:
        logger.error(f"Google Cloud Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Google Cloud Translation failed: {str(e)}")

async def translate_with_gemini(text: str) -> str:
    try:
        if not GEMINI_API_KEY:
            raise Exception("Gemini API key is missing")
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')  # Updated model name
        
        generation_config = {
            "temperature": 0.1  # Low temperature for accurate translation
        }
        
        prompt = f"""You are an expert translator specializing in professional development and coaching content. 

Task: Translate the following Persian text to English while maintaining:
1. Professional coaching terminology
2. Psychological concept accuracy
3. Motivational impact
4. Cultural context
5. Professional tone

Focus on EXACT translation with precise terminology matching. This is a translation task, not text generation.

Text to translate:
{text}

Provide only the direct translation, maintaining exact meaning."""
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if not hasattr(response, 'text'):
            raise Exception("No response from Gemini")
            
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Gemini translation error: {str(e)}")
        raise Exception(f"Gemini translation failed: {str(e)}")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard")
async def dashboard(request: Request):
    try:
        # Initialize default metrics structure
        default_metrics = {
            "today": {
                "translations": 0,
                "edits": 0,
                "success_rate": 0,
                "model_usage": {},
                "avg_quality_scores": {}
            },
            "this_week": {
                "translations": 0,
                "edits": 0,
                "success_rate": 0,
                "model_usage": {},
                "avg_quality_scores": {}
            },
            "total": {
                "translations": 0,
                "edits": 0,
                "success_rate": 0,
                "model_usage": {},
                "avg_quality_scores": {}
            }
        }

        # Get metrics with error handling
        try:
            model_metrics = stats.get_time_based_metrics()
            if not isinstance(model_metrics, dict):
                model_metrics = default_metrics
        except Exception as e:
            logger.error(f"Error getting time-based metrics: {str(e)}")
            model_metrics = default_metrics

        # Ensure all required data is present with default values
        dashboard_data = {
            "request": request,
            "total_translations": getattr(stats, 'total_translations', 0),
            "total_edits": getattr(stats, 'total_edits', 0),
            "active_users": len(getattr(stats, 'active_users', set())),
            "api_success_rate": round(getattr(stats, 'get_api_success_rate', lambda: 0.0)(), 1),
            "model_usage": dict(getattr(stats, 'model_usage', defaultdict(int))),
            "recent_activities": getattr(stats, 'recent_activities', []),
            "model_metrics": model_metrics
        }
        
        logger.info("Rendering dashboard with data")
        return templates.TemplateResponse("dashboard.html", dashboard_data)
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        # Return a simple error page instead of raising an exception
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "error": "An error occurred while loading the dashboard. Please try again later."
        })

def calculate_translation_quality_score(original_text: str, translated_text: str) -> int:
    score = 3  # Start with a neutral score
    
    # Check if the translation maintains the original meaning
    if len(translated_text.split()) > len(original_text.split()) * 0.8:
        score += 1
        
    # Check if professional terms are preserved
    professional_terms = ['coaching', 'leadership', 'development', 'skills', 'professional']
    term_count = sum(1 for term in professional_terms if term.lower() in translated_text.lower())
    if term_count >= 2:
        score += 1
        
    # Check if cultural context is preserved
    cultural_indicators = ['culture', 'language', 'social', 'professional']
    if any(indicator.lower() in translated_text.lower() for indicator in cultural_indicators):
        score += 1
    
    # Cap the score between 1 and 5
    return max(1, min(5, score))

@app.post("/translate")
async def translate(request: TranslationRequest):
    start_time = datetime.now()
    try:
        logger.info(f"Attempting to translate text using {request.model}: {request.text}")
        
        # Convert model string to ModelType
        try:
            model_type = ModelType(request.model)
        except ValueError:
            return {"error": f"Invalid model: {request.model}", "status_code": 400}
        
        if model_type == ModelType.GPT35:
            translated_text = await translate_with_openai(request.text, "gpt-3.5-turbo")
        elif model_type == ModelType.GPT4:
            translated_text = await translate_with_openai(request.text, "gpt-4")
        elif model_type == ModelType.CLAUDE:
            translated_text = await translate_with_claude(request.text)
        elif model_type == ModelType.GOOGLE:
            translated_text = await translate_with_google_cloud(request.text)
        elif model_type == ModelType.GEMINI:
            translated_text = await translate_with_gemini(request.text)
        else:
            raise HTTPException(status_code=400, detail="Invalid model specified")
            
        response_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Translation successful with {request.model}")
        
        # Calculate quality score for translation
        quality_score = calculate_translation_quality_score(request.text, translated_text)
        
        stats.add_translation(request.model, True, response_time, quality_score=quality_score)
        stats.record_request(True)
        return {"translated_text": translated_text}
        
    except Exception as e:
        response_time = (datetime.now() - start_time).total_seconds()
        error_type = type(e).__name__
        logger.error(f"Translation error: {str(e)}")
        stats.add_translation(request.model, False, response_time, error_type)
        stats.record_request(False)
        if isinstance(e, HTTPException):
            return {"error": e.detail, "status_code": e.status_code}
        return {"error": str(e), "status_code": 500}

def calculate_quality_score(original_text: str, edited_text: str, mode: str) -> int:
    score = 3  # Start with a neutral score
    
    # For Fast Mode
    if mode == "fast":
        # Check if the model didn't change too much (preserved your text)
        if len(edited_text.split()) > len(original_text.split()) * 0.95:
            score += 1
        # Check if proper punctuation was added
        if any(c in edited_text for c in ['.', '،', '؛', '؟']):
            score += 1
        # Check if the text has enough content
        if len(edited_text.split()) > 5:
            score += 1
            
    # For Detailed Mode
    else:
        # Check if professional terms were used correctly
        professional_terms = ['کوچینگ', 'رهبری', 'توسعه', 'مهارت', 'حرفه‌ای']
        term_count = sum(1 for term in professional_terms if term in edited_text)
        if term_count >= 2:
            score += 1
            
        # Check if the model maintained your content
        if len(edited_text.split()) > len(original_text.split()) * 0.9:
            score += 1
            
        # Check if cultural context was preserved
        cultural_indicators = ['فرهنگ', 'زبان', 'اجتماعی', 'حرفه‌ای']
        if any(indicator in edited_text for indicator in cultural_indicators):
            score += 1
    
    # Cap the score between 1 and 5
    return max(1, min(5, score))

@app.post("/edit")
async def edit(request: EditRequest):
    start_time = datetime.now()
    try:
        logger.info(f"Attempting to edit text in {request.mode} mode using {request.model}: {request.text}")
        
        # Convert model string to ModelType
        try:
            model_type = ModelType(request.model)
        except ValueError:
            logger.error(f"Invalid model type received: {request.model}")
            return {"error": f"Invalid model: {request.model}", "status_code": 400}
        
        # Detect language
        language = "Persian" if any("\u0600" <= c <= "\u06FF" for c in request.text) else "English"
        
        system_prompt = get_system_prompt(request.mode, language)
        user_prompt_edit = get_edit_prompt(request.text, request.mode, language)
        
        edited_text = ""
        explanation = ""
        
        if model_type == ModelType.GPT35:
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API key is missing")
                
            # Set temperature based on mode
            temperature = 0.5 if request.mode == "detailed" else 0.3

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # First get the edited text
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_edit}
                ],
                "temperature": temperature
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            edited_text = result["choices"][0]["message"]["content"].strip()
            
            # Then get the explanation
            user_prompt_explain = get_explanation_prompt(request.text, edited_text, language)
            data["messages"][1]["content"] = user_prompt_explain
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            explanation = result["choices"][0]["message"]["content"].strip()
            
        elif model_type == ModelType.GPT4:
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API key is missing")
                
            # Set temperature based on mode
            temperature = 0.4 if request.mode == "detailed" else 0.2

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # First get the edited text
            data = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_edit}
                ],
                "temperature": temperature
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            edited_text = result["choices"][0]["message"]["content"].strip()
            
            # Then get the explanation
            user_prompt_explain = get_explanation_prompt(request.text, edited_text, language)
            data["messages"][1]["content"] = user_prompt_explain
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            explanation = result["choices"][0]["message"]["content"].strip()
            
        elif model_type == ModelType.CLAUDE:
            if not ANTHROPIC_API_KEY:
                raise HTTPException(status_code=500, detail="Anthropic API key is missing")
                
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": ANTHROPIC_API_KEY
            }
            
            # First get the edited text
            data = {
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "temperature": 0.4 if request.mode == "detailed" else 0.2,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt_edit
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            edited_text = result["content"][0]["text"].strip()
            
            # Then get the explanation
            user_prompt_explain = get_explanation_prompt(request.text, edited_text, language)
            data["messages"][1]["content"] = user_prompt_explain
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            explanation = result["content"][0]["text"].strip()
            
        elif model_type == ModelType.GEMINI:
            if not GEMINI_API_KEY:
                raise HTTPException(status_code=500, detail="Gemini API key is missing")
                
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-pro')  # Updated model name
            
            generation_config = {
                "temperature": 0.25 if request.mode == "detailed" else 0.2
            }
            
            # First get the edited text
            prompt = f"{system_prompt}\n\n{user_prompt_edit}"
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not hasattr(response, 'text'):
                raise Exception("No response from Gemini")
            edited_text = response.text.strip()
            
            # Then get the explanation
            user_prompt_explain = get_explanation_prompt(request.text, edited_text, language)
            prompt = f"{system_prompt}\n\n{user_prompt_explain}"
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not hasattr(response, 'text'):
                raise Exception("No explanation from Gemini")
            explanation = response.text.strip()
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
        
        response_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Edit successful with {request.model}")
        
        # Calculate quality score
        quality_score = calculate_quality_score(request.text, edited_text, request.mode)
        
        stats.add_edit(request.model, request.mode, True, response_time, quality_score=quality_score)
        stats.record_request(True)
        return {"edited_text": edited_text, "explanation": explanation}
            
    except Exception as e:
        response_time = (datetime.now() - start_time).total_seconds()
        error_type = type(e).__name__
        logger.error(f"Edit error: {str(e)}")
        stats.add_edit(request.model, request.mode, False, response_time, error_type)
        stats.record_request(False)
        if isinstance(e, HTTPException):
            return {"error": e.detail, "status_code": e.status_code}
        return {"error": str(e), "status_code": 500}

async def test_gemini_connection() -> bool:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Hello")
        return hasattr(response, 'text')
    except Exception as e:
        logger.error(f"Error testing Gemini connection: {str(e)}")
        return False

# Add a route to test the connection
@app.get("/test-gemini")
async def test_gemini():
    """Endpoint to test Gemini API connection"""
    success = await test_gemini_connection()
    if success:
        return {"status": "success", "message": "Gemini API connection test passed"}
    else:
        raise HTTPException(status_code=500, detail="Gemini API connection test failed")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Test Gemini connection before starting the server
    asyncio.run(test_gemini_connection())
    
    uvicorn.run(app, host="0.0.0.0", port=8088)