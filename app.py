import time
from flask import Flask, request, jsonify
from neo4j import GraphDatabase
from google import genai
from google.genai import types
import json

app = Flask(__name__)

# Note: Ensure your environment/credentials are set correctly. 
# Using provided credentials from user snippet.
neo4j_driver = GraphDatabase.driver("neo4j+s://55115469.databases.neo4j.io", auth=("neo4j", "tTe604uH7jxZ-ZFyk7Jr6F19rqGqVxgBpZpi6Nfs8HA"))
gemini_client = genai.Client(api_key="AIzaSyDmjQ75Ok-H12IbR8ykmIwNRL2tqIYPHaE")

with open('graph-structure.json', 'r') as f:
    schema_data = json.load(f)

def format_schema_for_llm(data):
    schema_summary = "GRAPH DATABASE SCHEMA:\n"
    for item in data:
        label = item['label']
        details = item['details']
        if details['type'] == 'node':
            props = list(details.get('properties', {}).keys())
            rels = list(details.get('relationships', {}).keys())
            schema_summary += f"- Node [{label}]: Properties {props} | Relationships {rels}\n"
        else:
            schema_summary += f"- Relationship [{label}]: Properties {list(details.get('properties', {}).keys())}\n"
    return schema_summary

def execute_cypher(query):
    start_time = time.time()
    try:
        with neo4j_driver.session() as session:
            result = session.run(query)
            data = [record.data() for record in result]
            return data, None, (time.time() - start_time)
    except Exception as e:
        return None, str(e), (time.time() - start_time)

GRAPH_CONTEXT = format_schema_for_llm(schema_data)

SYSTEM_PROMPT = f"""
Act as a Senior Graph Data Scientist. You analyze career paths using a specific graph dataset.

SCHEMA CONTEXT:
{GRAPH_CONTEXT}

CORE OPERATIONAL RULES:
1. MANDATORY CLARIFICATION: On the VERY FIRST interaction of any session, you are FORBIDDEN from generating Cypher. Your ONLY permitted output is a single, targeted clarifying question to confirm the user's intent or define the scope of 'popularity' (e.g., by volume, by salary, or by recent trends).
2. FUZZY MATCHING: Never use exact equality for names. Use `toLower(n.name) CONTAINS toLower('search_term')`.
3. RELAXATION STRATEGY: 
   - Hop 1: Try to find the specific role requested.
   - Hop 2 (If Hop 1 is empty): Broaden the search. If 'Fullstack Software Engineer' failed, search for 'Fullstack' or 'Software Engineer'.
   - Hop 3 (If still empty): Search for the most common transitions in the general domain (e.g., all Engineering roles).
4. INSIGHT OVER ADHERENCE: If the specific data doesn't exist, pivot to the "next best" insight. Tell the user: "I couldn't find X, but here is what the data shows for the broader category of Y."
5. LIMITATIONS & SAMPLE SIZE: Always state the sample size (n=X) and the percentage of the total dataset this represents. Mention if the data is skewed.

CYPHER GUIDELINES:
- Use :Professional -[:HAS_EXPERIENCE]-> :Experience -[:ROLE_WAS]-> :JobTitle
- Transitions: (e1:Experience)-[:FOLLOWED_BY]->(e2:Experience)
- Always check relationship directions as per the schema.
"""

@app.route('/ask', methods=['POST'])
def handle_query():
    overall_start = time.time()
    data = request.json
    user_question = data.get('question', '')
    history_data = data.get('history', []) 
    query_budget = data.get('currentHop',0)
    
    print(f"\n🚀 [NEW QUERY]: {user_question}")

    messages = []
    
    for entry in history_data:
        messages.append(types.Content(role=entry['role'], parts=[types.Part(text=entry['text'])]))
    
    messages.append(types.Content(role="user", parts=[types.Part(text=user_question)]))

    if len(history_data) == 0:
        print("❓ [CLARIFYING]: First contact - requesting clarification.")
        response = gemini_client.models.generate_content(
            model="gemma-3-12b", 
            contents=messages,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
        )
        return jsonify({"question": response.text,
        "currentHop":0})

    while query_budget < 3:
        hop_id = query_budget + 1
        print(f"hop id: {hop_id}")
        hop_context = f"This is HOP {hop_id}/3. Generate a Cypher query or type 'ANALYZE' if you have enough data."
        
        response = gemini_client.models.generate_content(
            model="gemma-3-12b",
            contents=messages + [types.Content(role="user", parts=[types.Part(text=hop_context)])],
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
        )
        
        decision = response.text
        if "```cypher" not in decision:
            print(f"✅ [HOP {hop_id}]: LLM decided to ANALYZE.")
            break

        cypher_query = decision.split("```cypher")[1].split("```")[0].strip()
        print(f"🔍 [HOP {hop_id}]: Executing Cypher...")
        
        db_results, error, db_time = execute_cypher(cypher_query)
        print(f"query: {cypher_query}, result: {db_results}")

        if error:
            feedback = f"SYSTEM ERROR: Your Cypher failed: {error}. Fix syntax and try again."
            messages.append(types.Content(role="model", parts=[types.Part(text=decision)]))
            messages.append(types.Content(role="user", parts=[types.Part(text=feedback)]))
            print(f"❌ [DB ERROR]: {error}")
            continue

        query_budget += 1
        messages.append(types.Content(role="model", parts=[types.Part(text=decision)]))
        
        if not db_results:
            feedback = f"HOP {hop_id} RESULT: 0 records found. The search term may be too specific. Broaden your scope for HOP {hop_id+1}."
            print(f"⚠️ [EMPTY RESULT]: Feedback sent to LLM.")
        else:
            feedback = f"HOP {hop_id} RESULT: Found {len(db_results)} records. Data: {db_results}"
            print(f"📊 [SUCCESS]: {len(db_results)} records found in {db_time:.4f}s")
            
        messages.append(types.Content(role="user", parts=[types.Part(text=feedback)]))

    print(f"🧠 [SYNTHESIZING]: Final response generation...")
    final_response = gemini_client.models.generate_content(
        model="gemma-3-12b",
        contents=messages + [types.Content(role="user", parts=[types.Part(text="Synthesize the final answer based on all gathered data")])],
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    )
    
    print(f"🏁 [COMPLETE]: Finished in {time.time() - overall_start:.2f}s\n" + "-"*30)
    return jsonify({"analysis": final_response.text})

if __name__ == '__main__':
    app.run(port=5000, debug=True)