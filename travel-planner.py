import os
from dotenv import load_dotenv
from autogen import ConversableAgent, GroupChat, GroupChatManager

load_dotenv()

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ]
}

user_proxy = ConversableAgent(
    name="User_Proxy_Agent",
    system_message="You are a user proxy agent.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

destination_expert = ConversableAgent(
    name="Destination_Expert_Agent",
    system_message="""
    You are the Destination Expert, a specialist in global travel destinations. Your responsibilities include:
    1. Analyzing user preferences (e.g., climate, activities, culture) to suggest suitable destinations.
    2. Providing detailed information about recommended locations, including attractions, best times to visit, and local customs.
    3. Consider factors like seasonality, events, and travel advisories in your recommendations.
    4. SELECT 1 destination that you think is the best choice
    5. DO NOT GIVE AN ITINERARY
    Base your suggestions on a wide range of global destinations and current travel trends.
    Format your response with a clear "DESTINATION SUMMARY" header.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

itinerary_creator = ConversableAgent(
    name="Itinerary_Creator_Agent",
    system_message="""
    You are the Itinerary Creator, responsible for crafting detailed travel itineraries. Your role involves:
    1. Creating day-by-day schedules for the entire trip duration.
    2. Incorporating user preferences for activities, pace of travel, and must-see attractions.
    3. Balancing tourist attractions with local experiences.
    4. Considering practical aspects like travel times, opening hours, and meal times.
    5. Ensuring the itinerary aligns with the budget provided by the BudgetAnalyst.
    Aim to create engaging, well-paced itineraries that maximize the travel experience within given constraints.
    Format your response with a clear "ITINERARY" header, followed by a day-by-day breakdown.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

budget_analyst = ConversableAgent(
    name="Budget_Analyst_Agent",
    system_message="""
    You are the Budge tAnalyst, an expert in travel budgeting and financial planning. Your tasks are to:
    1. Analyze the user's overall budget for the trip.
    2. Provide detailed cost estimates for various aspects of the trip (transportation, accommodation, food, activities).
    3. Suggest ways to optimize spending and find cost-effective options.
    4. Create a budget breakdown for the entire trip.
    5. Offer financial advice related to travel (e.g., best payment methods, travel insurance).
    Always strive for accuracy in your estimates and provide practical financial advice for travelers.
    Format your response with a clear "BUDGET" header.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

report_writer = ConversableAgent(
    name="Report_Writer_Agent",
    system_message="""
    You are the Report Compiler agent, tasked with creating a comprehensive travel report based on the outputs of the following agents:

    Destination_Expert_Agent: Provides a recommended destination.
    Itinerary_Creator_Agent: Creates a detailed itinerary for the chosen destination.
    Budget_Analyst_Agent: Analyzes the budget and provides cost estimates for the trip.

    Your Task:

    Gather Information: Collect the outputs from each agent.
    Structure the Report: Organize the information into a clear and concise report format.
    Create a Summary: Provide a brief overview of the recommended destination, itinerary, and budget.
    Highlight Key Points: Emphasize the most important aspects of the trip, such as unique experiences or cost-saving tips.

    Report Structure:

    Introduction: A brief welcome and overview of the trip.
    Destination Summary: Details from the Destination_Expert_Agent, including the chosen destination and reasons for recommendation.
    Cultural Tips: Information on local customs, etiquette, and cultural norms.
    Itinerary: The detailed itinerary created by the Itinerary_Creator_Agent, including day-by-day activities and accommodations.
    Transportation: A summary of transport modes found at the destination and their prices
    Budget Breakdown: The cost estimates and financial advice from the Budget_Analyst_Agent.
    Packing List: List of essential and optional items to pack for your trip.
    Conclusion: A summary of the trip and any final recommendations or suggestions.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

allowed_transitions = {
    user_proxy: [destination_expert, user_proxy],
    destination_expert: [itinerary_creator, user_proxy],
    itinerary_creator: [budget_analyst, user_proxy],
    budget_analyst: [report_writer, user_proxy],
    report_writer: [user_proxy],
}

group_chat = GroupChat(
    agents=[
        user_proxy,
        destination_expert,
        itinerary_creator,
        budget_analyst,
        report_writer,
    ],
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed",
    messages=[],
    max_round=6,
)

travel_planner_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)


def main():
    print("Welcome to the Travel Planning Assistant!")
    print("You can plan your trip by providing details about your desired vacation.")
    print("Type 'exit', 'quit', or 'bye' to end the session.")

    while True:
        user_input = input(f"Enter your trip details: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print(f"Goodbye! Have a great day!!")
            break

        print("\nPlanning your trip... This may take a moment.")

        try:
            chat_result = user_proxy.initiate_chat(
                travel_planner_manager,
                message=user_input,
                summary_method="last_msg",
            )

            report = next(
                msg["content"]
                for msg in chat_result.chat_history
                if msg.get("name") == "Report_Writer_Agent"
            )

            print("\n-------------Final Holiday Details----------\n")
            print(report)
            print("\n--------------------------------------------\n")

        except Exception as e:
            print(f"An error occurred while planning your trip: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()
