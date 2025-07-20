import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Data structures for storing user information and results
@dataclass
class UserData:
    weight: float = 0.0
    height: float = 0.0
    age: int = 0
    gender: str = ""
    dietary_preference: str = ""

@dataclass
class BMIResult:
    bmi_value: float
    category: str
    health_status: str

@dataclass
class HealthPlan:
    user_data: UserData
    bmi_result: BMIResult
    diet_plan: Dict[str, Any]
    workout_plan: Dict[str, Any]

# Base Agent Class
class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.conversation_history = []
    
    def log_message(self, message: str):
        self.conversation_history.append(f"[{self.name}]: {message}")
        print(f"[{self.name}]: {message}")
    
    @abstractmethod
    def process(self, *args, **kwargs):
        pass

# BMI Calculation Tool
class BMITool:
    @staticmethod
    def calculate_bmi(weight: float, height_cm: float) -> float:
        """
        Calculate BMI using weight (kg) and height (cm)
        Formula: BMI = Weight (kg) / (Height (m))^2
        """
        # Convert height from cm to meters
        height_m = height_cm / 100
        
        # Calculate BMI
        bmi = weight / (height_m ** 2)
        return round(bmi, 2)
    
    @staticmethod
    def get_bmi_category(bmi: float) -> Tuple[str, str]:
        """Return BMI category and health status"""
        if bmi < 18.5:
            return "Underweight", "Below normal weight range"
        elif 18.5 <= bmi < 25:
            return "Normal", "Healthy weight range"
        elif 25 <= bmi < 30:
            return "Overweight", "Above normal weight range"
        else:
            return "Obese", "Significantly above normal weight range"

# User Proxy Agent - Collects user data
class UserProxyAgent(BaseAgent):
    def __init__(self):
        super().__init__("User Proxy Agent")
        self.user_data = UserData()
    
    def collect_user_input(self) -> UserData:
        """Collect user inputs interactively"""
        self.log_message("Welcome to the Smart Health Assistant! I'll collect your information.")
        
        try:
            self.user_data.weight = float(input("Enter your weight (in kg): "))
            self.user_data.height = float(input("Enter your height (in cm): "))
            self.user_data.age = int(input("Enter your age: "))
            
            while True:
                gender = input("Enter your gender (Male/Female/Other): ").strip().title()
                if gender in ['Male', 'Female', 'Other']:
                    self.user_data.gender = gender
                    break
                print("Please enter Male, Female, or Other")
            
            while True:
                diet_pref = input("Enter your dietary preference (Veg/Non-Veg/Vegan): ").strip().title()
                if diet_pref in ['Veg', 'Non-Veg', 'Vegan']:
                    self.user_data.dietary_preference = diet_pref
                    break
                print("Please enter Veg, Non-Veg, or Vegan")
            
            self.log_message(f"Collected user data: Weight={self.user_data.weight}kg, "
                           f"Height={self.user_data.height}cm, Age={self.user_data.age}, "
                           f"Gender={self.user_data.gender}, Diet={self.user_data.dietary_preference}")
            
            return self.user_data
        
        except ValueError as e:
            self.log_message(f"Invalid input: {e}")
            return self.collect_user_input()
    
    def process(self) -> UserData:
        return self.collect_user_input()

# BMI Agent - Analyzes BMI and provides health recommendations
class BMIAgent(BaseAgent):
    def __init__(self):
        super().__init__("BMI Agent")
        self.bmi_tool = BMITool()
    
    def analyze_bmi(self, user_data: UserData) -> BMIResult:
        """Calculate BMI and provide health recommendations"""
        self.log_message(f"Analyzing BMI for user with weight {user_data.weight}kg and height {user_data.height}cm")
        
        # Calculate BMI using the registered tool
        bmi_value = self.bmi_tool.calculate_bmi(user_data.weight, user_data.height)
        category, health_status = self.bmi_tool.get_bmi_category(bmi_value)
        
        bmi_result = BMIResult(bmi_value, category, health_status)
        
        self.log_message(f"BMI Analysis Complete: BMI = {bmi_value}, Category = {category}")
        self.log_message(f"Health Status: {health_status}")
        
        # Provide specific recommendations based on BMI category
        if category == "Underweight":
            recommendation = "Focus on gaining healthy weight through balanced nutrition and strength training."
        elif category == "Normal":
            recommendation = "Maintain your current weight with balanced diet and regular exercise."
        elif category == "Overweight":
            recommendation = "Consider gradual weight loss through caloric deficit and increased physical activity."
        else:  # Obese
            recommendation = "Prioritize weight loss through significant dietary changes and structured exercise program."
        
        self.log_message(f"Recommendation: {recommendation}")
        return bmi_result
    
    def process(self, user_data: UserData) -> BMIResult:
        return self.analyze_bmi(user_data)

# Diet Planner Agent - Creates personalized meal plans
class DietPlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Diet Planner Agent")
    
    def create_meal_plan(self, user_data: UserData, bmi_result: BMIResult) -> Dict[str, Any]:
        """Create personalized meal plan based on BMI and dietary preferences"""
        self.log_message(f"Creating meal plan for {bmi_result.category} individual with {user_data.dietary_preference} preference")
        
        # Base meal structure
        meal_plan = {
            "daily_calories": self._calculate_daily_calories(user_data, bmi_result),
            "meals": {},
            "dietary_focus": self._get_dietary_focus(bmi_result.category),
            "hydration": "Drink 8-10 glasses of water daily"
        }
        
        # Create meal plans based on dietary preference
        if user_data.dietary_preference == "Veg":
            meal_plan["meals"] = self._create_vegetarian_meals(bmi_result.category)
        elif user_data.dietary_preference == "Vegan":
            meal_plan["meals"] = self._create_vegan_meals(bmi_result.category)
        else:  # Non-Veg
            meal_plan["meals"] = self._create_non_vegetarian_meals(bmi_result.category)
        
        self.log_message(f"Meal plan created with {meal_plan['daily_calories']} daily calories")
        self._log_meal_summary(meal_plan)
        
        return meal_plan
    
    def _calculate_daily_calories(self, user_data: UserData, bmi_result: BMIResult) -> int:
        """Calculate recommended daily calories based on BMR and BMI category"""
        # Basic BMR calculation (simplified)
        if user_data.gender.lower() == "male":
            bmr = 88.362 + (13.397 * user_data.weight) + (4.799 * user_data.height) - (5.677 * user_data.age)
        else:
            bmr = 447.593 + (9.247 * user_data.weight) + (3.098 * user_data.height) - (4.330 * user_data.age)
        
        # Adjust based on BMI category
        if bmi_result.category == "Underweight":
            return int(bmr * 1.6)  # Increase calories for weight gain
        elif bmi_result.category == "Normal":
            return int(bmr * 1.4)  # Maintain current weight
        elif bmi_result.category == "Overweight":
            return int(bmr * 1.2)  # Moderate deficit
        else:  # Obese
            return int(bmr * 1.1)  # Larger deficit for weight loss
    
    def _get_dietary_focus(self, bmi_category: str) -> str:
        focus_map = {
            "Underweight": "High protein, healthy fats, complex carbohydrates",
            "Normal": "Balanced macronutrients, variety of nutrients",
            "Overweight": "High protein, high fiber, controlled portions",
            "Obese": "Very high protein, high fiber, low glycemic index foods"
        }
        return focus_map.get(bmi_category, "Balanced nutrition")
    
    def _create_vegetarian_meals(self, bmi_category: str) -> Dict[str, List[str]]:
        base_meals = {
            "breakfast": ["Oats with fruits and nuts", "Greek yogurt with berries", "Whole grain toast with avocado"],
            "lunch": ["Quinoa salad with vegetables", "Lentil curry with brown rice", "Vegetable soup with whole grain bread"],
            "dinner": ["Grilled paneer with vegetables", "Bean and vegetable stir-fry", "Vegetable curry with quinoa"],
            "snacks": ["Mixed nuts", "Fruit with yogurt", "Vegetable sticks with hummus"]
        }
        
        if bmi_category in ["Overweight", "Obese"]:
            # Add more fiber-rich and lower-calorie options
            base_meals["lunch"].append("Large mixed salad with chickpeas")
            base_meals["dinner"].append("Steamed vegetables with tofu")
        
        return base_meals
    
    def _create_vegan_meals(self, bmi_category: str) -> Dict[str, List[str]]:
        base_meals = {
            "breakfast": ["Oat smoothie with plant milk and fruits", "Chia seed pudding", "Avocado toast on whole grain bread"],
            "lunch": ["Buddha bowl with quinoa and vegetables", "Lentil soup with vegetables", "Chickpea salad wrap"],
            "dinner": ["Tofu stir-fry with brown rice", "Black bean curry", "Roasted vegetables with quinoa"],
            "snacks": ["Mixed nuts and seeds", "Fresh fruits", "Vegetable sticks with tahini"]
        }
        
        if bmi_category in ["Overweight", "Obese"]:
            base_meals["lunch"].append("Large kale salad with hemp seeds")
            base_meals["dinner"].append("Steamed vegetables with tempeh")
        
        return base_meals
    
    def _create_non_vegetarian_meals(self, bmi_category: str) -> Dict[str, List[str]]:
        base_meals = {
            "breakfast": ["Scrambled eggs with vegetables", "Greek yogurt with nuts", "Oatmeal with protein powder"],
            "lunch": ["Grilled chicken salad", "Fish curry with brown rice", "Turkey and vegetable soup"],
            "dinner": ["Baked salmon with vegetables", "Lean beef stir-fry", "Grilled chicken with quinoa"],
            "snacks": ["Hard-boiled eggs", "Greek yogurt", "Mixed nuts"]
        }
        
        if bmi_category in ["Overweight", "Obese"]:
            base_meals["lunch"].append("Large protein salad with grilled chicken")
            base_meals["dinner"].append("Steamed fish with vegetables")
        
        return base_meals
    
    def _log_meal_summary(self, meal_plan: Dict[str, Any]):
        self.log_message(f"Dietary Focus: {meal_plan['dietary_focus']}")
        for meal_type, options in meal_plan['meals'].items():
            self.log_message(f"{meal_type.title()} options: {len(options)} varieties available")
    
    def process(self, user_data: UserData, bmi_result: BMIResult) -> Dict[str, Any]:
        return self.create_meal_plan(user_data, bmi_result)

# Workout Scheduler Agent - Creates personalized workout plans
class WorkoutSchedulerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Workout Scheduler Agent")
    
    def create_workout_plan(self, user_data: UserData, diet_plan: Dict[str, Any], bmi_result: BMIResult) -> Dict[str, Any]:
        """Create personalized weekly workout plan"""
        self.log_message(f"Creating workout plan for {user_data.age}-year-old {user_data.gender} with {bmi_result.category} BMI")
        
        workout_plan = {
            "weekly_schedule": {},
            "workout_focus": self._get_workout_focus(bmi_result.category),
            "intensity": self._get_intensity_level(user_data.age, bmi_result.category),
            "duration_per_session": self._get_session_duration(bmi_result.category),
            "rest_days": 2,
            "additional_notes": self._get_additional_notes(user_data, bmi_result)
        }
        
        # Create weekly schedule
        workout_plan["weekly_schedule"] = self._create_weekly_schedule(user_data, bmi_result)
        
        self.log_message(f"Workout plan created with {workout_plan['intensity']} intensity")
        self._log_workout_summary(workout_plan)
        
        return workout_plan
    
    def _get_workout_focus(self, bmi_category: str) -> str:
        focus_map = {
            "Underweight": "Strength training and muscle building",
            "Normal": "Overall fitness and maintenance",
            "Overweight": "Cardio and strength training for weight loss",
            "Obese": "Low-impact cardio and gradual strength building"
        }
        return focus_map.get(bmi_category, "General fitness")
    
    def _get_intensity_level(self, age: int, bmi_category: str) -> str:
        if age > 50 or bmi_category == "Obese":
            return "Low to Moderate"
        elif age > 35 or bmi_category == "Overweight":
            return "Moderate"
        else:
            return "Moderate to High"
    
    def _get_session_duration(self, bmi_category: str) -> str:
        duration_map = {
            "Underweight": "45-60 minutes",
            "Normal": "45-60 minutes",
            "Overweight": "45-75 minutes",
            "Obese": "30-45 minutes"
        }
        return duration_map.get(bmi_category, "45-60 minutes")
    
    def _get_additional_notes(self, user_data: UserData, bmi_result: BMIResult) -> List[str]:
        notes = []
        
        if user_data.age > 50:
            notes.append("Focus on low-impact exercises and joint-friendly movements")
        
        if bmi_result.category == "Obese":
            notes.append("Start with low-impact activities and gradually increase intensity")
            notes.append("Consider water-based exercises to reduce joint stress")
        
        if bmi_result.category == "Underweight":
            notes.append("Combine strength training with adequate rest for muscle growth")
            notes.append("Don't overdo cardio - focus on building muscle mass")
        
        notes.append("Always warm up before exercising and cool down afterwards")
        notes.append("Stay hydrated throughout your workouts")
        notes.append("Listen to your body and rest when needed")
        
        return notes
    
    def _create_weekly_schedule(self, user_data: UserData, bmi_result: BMIResult) -> Dict[str, Dict[str, Any]]:
        schedule = {}
        
        if bmi_result.category == "Underweight":
            schedule = {
                "Monday": {"type": "Strength Training", "focus": "Upper Body", "duration": "45-60 min"},
                "Tuesday": {"type": "Active Recovery", "focus": "Light Walking/Yoga", "duration": "30 min"},
                "Wednesday": {"type": "Strength Training", "focus": "Lower Body", "duration": "45-60 min"},
                "Thursday": {"type": "Cardio", "focus": "Moderate Intensity", "duration": "30 min"},
                "Friday": {"type": "Strength Training", "focus": "Full Body", "duration": "45-60 min"},
                "Saturday": {"type": "Active Recovery", "focus": "Stretching/Light Activity", "duration": "30 min"},
                "Sunday": {"type": "Rest", "focus": "Complete Rest", "duration": "0 min"}
            }
        elif bmi_result.category == "Normal":
            schedule = {
                "Monday": {"type": "Strength Training", "focus": "Upper Body", "duration": "45 min"},
                "Tuesday": {"type": "Cardio", "focus": "Moderate to High Intensity", "duration": "45 min"},
                "Wednesday": {"type": "Strength Training", "focus": "Lower Body", "duration": "45 min"},
                "Thursday": {"type": "Active Recovery", "focus": "Yoga/Stretching", "duration": "30 min"},
                "Friday": {"type": "Full Body Workout", "focus": "Functional Training", "duration": "60 min"},
                "Saturday": {"type": "Cardio", "focus": "Fun Activity (Sports/Dancing)", "duration": "60 min"},
                "Sunday": {"type": "Rest", "focus": "Complete Rest", "duration": "0 min"}
            }
        elif bmi_result.category == "Overweight":
            schedule = {
                "Monday": {"type": "Cardio", "focus": "Moderate Intensity", "duration": "45-60 min"},
                "Tuesday": {"type": "Strength Training", "focus": "Upper Body", "duration": "45 min"},
                "Wednesday": {"type": "Cardio", "focus": "Interval Training", "duration": "45 min"},
                "Thursday": {"type": "Strength Training", "focus": "Lower Body", "duration": "45 min"},
                "Friday": {"type": "Cardio", "focus": "Steady State", "duration": "60 min"},
                "Saturday": {"type": "Active Recovery", "focus": "Walking/Light Activity", "duration": "30-45 min"},
                "Sunday": {"type": "Rest", "focus": "Complete Rest", "duration": "0 min"}
            }
        else:  # Obese
            schedule = {
                "Monday": {"type": "Low-Impact Cardio", "focus": "Walking/Swimming", "duration": "30-45 min"},
                "Tuesday": {"type": "Strength Training", "focus": "Light Weights/Bodyweight", "duration": "30 min"},
                "Wednesday": {"type": "Low-Impact Cardio", "focus": "Stationary Bike/Elliptical", "duration": "30-45 min"},
                "Thursday": {"type": "Active Recovery", "focus": "Gentle Stretching", "duration": "20-30 min"},
                "Friday": {"type": "Low-Impact Cardio", "focus": "Water Exercises/Walking", "duration": "30-45 min"},
                "Saturday": {"type": "Strength Training", "focus": "Full Body Light", "duration": "30 min"},
                "Sunday": {"type": "Rest", "focus": "Complete Rest", "duration": "0 min"}
            }
        
        return schedule
    
    def _log_workout_summary(self, workout_plan: Dict[str, Any]):
        self.log_message(f"Workout Focus: {workout_plan['workout_focus']}")
        self.log_message(f"Session Duration: {workout_plan['duration_per_session']}")
        self.log_message(f"Weekly Schedule: 5 active days, 2 rest days")
    
    def process(self, user_data: UserData, diet_plan: Dict[str, Any], bmi_result: BMIResult) -> Dict[str, Any]:
        return self.create_workout_plan(user_data, diet_plan, bmi_result)

# Main Health Assistant Orchestrator
class SmartHealthAssistant:
    def __init__(self):
        self.user_proxy = UserProxyAgent()
        self.bmi_agent = BMIAgent()
        self.diet_planner = DietPlannerAgent()
        self.workout_scheduler = WorkoutSchedulerAgent()
        self.health_plan = None
    
    def run_health_assessment(self) -> HealthPlan:
        """Run the complete health assessment process"""
        print("="*60)
        print("🏥 SMART HEALTH ASSISTANT - SEQUENTIAL AGENT SYSTEM 🏥")
        print("="*60)
        
        try:
            # Step 1: Collect user data
            print("\n" + "="*50)
            print("STEP 1: USER DATA COLLECTION")
            print("="*50)
            user_data = self.user_proxy.process()
            
            # Step 2 & 3: Calculate BMI and get health recommendations
            print("\n" + "="*50)
            print("STEP 2-3: BMI ANALYSIS AND HEALTH RECOMMENDATIONS")
            print("="*50)
            bmi_result = self.bmi_agent.process(user_data)
            
            # Step 4: Create diet plan
            print("\n" + "="*50)
            print("STEP 4: PERSONALIZED DIET PLANNING")
            print("="*50)
            diet_plan = self.diet_planner.process(user_data, bmi_result)
            
            # Step 5: Create workout plan
            print("\n" + "="*50)
            print("STEP 5: PERSONALIZED WORKOUT SCHEDULING")
            print("="*50)
            workout_plan = self.workout_scheduler.process(user_data, diet_plan, bmi_result)
            
            # Create comprehensive health plan
            self.health_plan = HealthPlan(user_data, bmi_result, diet_plan, workout_plan)
            
            # Display final results
            self._display_final_health_plan()
            
            return self.health_plan
            
        except Exception as e:
            print(f"Error during health assessment: {e}")
            return None
    
    def _display_final_health_plan(self):
        """Display the complete health plan in a formatted manner"""
        print("\n" + "="*60)
        print("🌟 YOUR PERSONALIZED HEALTH PLAN 🌟")
        print("="*60)
        
        if not self.health_plan:
            print("No health plan available.")
            return
        
        # User Summary
        print("\n📋 USER PROFILE:")
        print("-" * 30)
        print(f"Age: {self.health_plan.user_data.age} years")
        print(f"Gender: {self.health_plan.user_data.gender}")
        print(f"Weight: {self.health_plan.user_data.weight} kg")
        print(f"Height: {self.health_plan.user_data.height} cm")
        print(f"Dietary Preference: {self.health_plan.user_data.dietary_preference}")
        
        # BMI Analysis
        print("\n📊 BMI ANALYSIS:")
        print("-" * 30)
        print(f"BMI Value: {self.health_plan.bmi_result.bmi_value}")
        print(f"Category: {self.health_plan.bmi_result.category}")
        print(f"Health Status: {self.health_plan.bmi_result.health_status}")
        
        # Diet Plan
        print("\n🍽️ DIET PLAN:")
        print("-" * 30)
        print(f"Daily Calories: {self.health_plan.diet_plan['daily_calories']} kcal")
        print(f"Dietary Focus: {self.health_plan.diet_plan['dietary_focus']}")
        print(f"Hydration: {self.health_plan.diet_plan['hydration']}")
        
        print("\nMeal Options:")
        for meal_type, options in self.health_plan.diet_plan['meals'].items():
            print(f"  {meal_type.title()}: {', '.join(options[:2])}...")  # Show first 2 options
        
        # Workout Plan
        print("\n🏋️ WORKOUT PLAN:")
        print("-" * 30)
        print(f"Focus: {self.health_plan.workout_plan['workout_focus']}")
        print(f"Intensity: {self.health_plan.workout_plan['intensity']}")
        print(f"Session Duration: {self.health_plan.workout_plan['duration_per_session']}")
        print(f"Rest Days per Week: {self.health_plan.workout_plan['rest_days']}")
        
        print("\nWeekly Schedule:")
        for day, workout in self.health_plan.workout_plan['weekly_schedule'].items():
            print(f"  {day}: {workout['type']} - {workout['focus']} ({workout['duration']})")
        
        print("\n💡 ADDITIONAL NOTES:")
        print("-" * 30)
        for note in self.health_plan.workout_plan['additional_notes']:
            print(f"  • {note}")
        
        print("\n" + "="*60)
        print("✅ HEALTH ASSESSMENT COMPLETE!")
        print("Remember to consult with healthcare professionals before starting any new diet or exercise program.")
        print("="*60)

# Demo function to show the system working with sample data
def run_demo():
    """Run a demo with sample data"""
    print("🔄 RUNNING DEMO MODE WITH SAMPLE DATA")
    print("="*50)
    
    # Create sample user data
    sample_data = UserData(
        weight=70.0,
        height=175.0,
        age=28,
        gender="Male",
        dietary_preference="Non-Veg"
    )
    
    # Initialize agents
    bmi_agent = BMIAgent()
    diet_planner = DietPlannerAgent()
    workout_scheduler = WorkoutSchedulerAgent()
    
    # Process through agents
    bmi_result = bmi_agent.process(sample_data)
    diet_plan = diet_planner.process(sample_data, bmi_result)
    workout_plan = workout_scheduler.process(sample_data, diet_plan, bmi_result)
    
    # Create and display health plan
    health_plan = HealthPlan(sample_data, bmi_result, diet_plan, workout_plan)
    
    assistant = SmartHealthAssistant()
    assistant.health_plan = health_plan
    assistant._display_final_health_plan()

# Main execution
if __name__ == "__main__":
    # Ask user if they want to run demo or interactive mode
    mode = input("Choose mode: (1) Interactive Mode (2) Demo Mode: ").strip()
    
    if mode == "2":
        run_demo()
    else:
        # Run the interactive health assistant
        assistant = SmartHealthAssistant()
        health_plan = assistant.run_health_assessment()
        
        if health_plan:
            print("\n✅ Health assessment completed successfully!")
        else:
            print("\n❌ Health assessment failed. Please try again.")