import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Annotated

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymongo
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, add_messages, START

from RAG import RAGSystem

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
rag = RAGSystem(groq_api_key=groq_api_key)


@dataclass
class VizRecommendation:
    title: str
    description: str
    columns: List[str]
    reasoning: str
    plot_type: str
    filenames: List[str]
    metadata: Dict[str, Any] = None


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    file_names: List[str]
    user_request: Optional[str]
    data_loaded: bool
    data_analysis_complete: bool
    recommendations_generated: bool
    plots_created: bool
    current_dataframes: Dict[str, pd.DataFrame]
    viz_recommendations: List[Dict[str, Any]]
    generated_plots: List[Dict[str, Any]]
    error_state: Optional[str]
    next_action: str
    rag_context: Optional[str]


class AgenticVizSystem:
    def __init__(self, mongo_string: str, api_key: str, model_name: str = "gemini-pro", viz_mode: str = "auto"):
        self.viz_mode = viz_mode
        self.mongo_string = mongo_string
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(
            model_name=model_name,
            api_key=api_key,
            temperature=0.2
        )
        self.collection = None
        self.mongo_setup()
        self.build_agent_graph()

    def mongo_setup(self):
        try:
            client = pymongo.MongoClient(self.mongo_string)
            db = client["Dashboard_project"]
            self.collection = db["tests"]
        except Exception as e:
            print(f"MongoDB setup failed: {e}")

    def build_agent_graph(self):
        workflow = StateGraph(AgentState)

        # Creating Nodes
        workflow.add_node("planner", self.planner_agent)
        workflow.add_node("data_loader", self.data_loader_agent)
        workflow.add_node("data_analyzer", self.data_analyzer_agent)
        workflow.add_node("rag_consultant", self.rag_consultant_agent)
        workflow.add_node("viz_recommendation", self.viz_recommendation_agent)
        workflow.add_node("plot_generator", self.plot_generator_agent)
        workflow.add_node("quality_checker", self.quality_checker_agent)
        workflow.add_node("error_handler", self.error_handler_agent)

        workflow.add_edge(START, "planner")

        workflow.add_conditional_edges(
            "planner",
            self.route_from_planner,
            {
                "load_data": "data_loader",
                "error": "error_handler",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "data_loader",
            self.route_from_data_loader,
            {
                "analyze_data": "data_analyzer",
                "error": "error_handler"
            }
        )
        workflow.add_conditional_edges(
            "data_analyzer",
            self.route_from_analyzer,
            {
                "consult_rag": "rag_consultant",
                "generate_recommendations": "viz_recommendation",  # Fixed: was "viz_recommender"
                "error": "error_handler"
            }
        )
        workflow.add_edge("rag_consultant", "viz_recommendation")  # Fixed: was "viz_recommender"
        workflow.add_conditional_edges(
            "viz_recommendation",  # Fixed: was "viz_recommender"
            self.route_from_recommender,
            {
                "create_plots": "plot_generator",
                "error": "error_handler"
            }
        )
        workflow.add_conditional_edges(
            "plot_generator",
            self.route_from_plot_generator,
            {
                "quality_check": "quality_checker",
                "error": "error_handler"
            }
        )
        workflow.add_conditional_edges(
            "quality_checker",
            self.route_from_quality_checker,
            {
                "regenerate": "viz_recommendation",  # Fixed: was "viz_recommender"
                "complete": END,
                "error": "error_handler"
            }
        )

        workflow.add_edge("error_handler", END)
        self.graph = workflow.compile()

    def planner_agent(self, state: AgentState) -> AgentState:
        """Planning agent - decides the overall strategy"""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        planning_prompt = f"""
        You are a planning agent for data visualization. Analyze this request and decide the next action:

        Request: {last_message}
        Files available: {state.get('file_names', [])}
        User request: {state.get('user_request', 'None')}

        Based on this, what should we do next?
        - If files need to be loaded: respond "LOAD_DATA"
        - If there's an error: respond "ERROR: <description>"
        - If request is unclear: respond "ERROR: Request unclear"

        Respond with just the action.
        """
        response = self.llm.invoke([HumanMessage(content=planning_prompt)])
        decision = response.content.strip()

        if decision.startswith("ERROR"):
            state["error_state"] = decision
            state["next_action"] = "error"
        elif decision == "LOAD_DATA":
            state["next_action"] = "load_data"
        else:
            state["next_action"] = "end"

        state["messages"] = state["messages"] + [AIMessage(content=f"Planner decision: {decision}")]
        return state

    def data_loader_agent(self, state: AgentState) -> AgentState:
        """Data loading agent - loads and validates data"""
        file_names = state.get("file_names", [])

        if not file_names:
            state["error_state"] = "No files provided for loading"
            state["next_action"] = "error"
            return state

        dataframes = {}

        for file_name in file_names:
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_name)
                elif file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_name)
                else:
                    state["error_state"] = f"Unsupported file format: {file_name}"
                    state["next_action"] = "error"
                    return state

                if df.empty:
                    state["error_state"] = f"File {file_name} is empty"
                    state["next_action"] = "error"
                    return state

                key = os.path.splitext(os.path.basename(file_name))[0]
                dataframes[key] = df

            except Exception as e:
                state["error_state"] = f"Error loading {file_name}: {str(e)}"
                state["next_action"] = "error"
                return state

        state["current_dataframes"] = dataframes
        state["data_loaded"] = True
        state["next_action"] = "analyze_data"
        state["messages"] = state["messages"] + [AIMessage(content=f"Successfully loaded {len(dataframes)} datasets")]
        return state

    def data_analyzer_agent(self, state: AgentState) -> AgentState:
        """Data analysis agent - analyzes loaded data"""
        dataframes = state.get("current_dataframes", {})

        if not dataframes:
            state["error_state"] = "No dataframes available for analysis"
            state["next_action"] = "error"
            return state

        analysis = {}

        # Fixed: Changed from 'dataframes' to 'dataframes.items()'
        for name, df in dataframes.items():
            analysis[name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head(3).to_dict()
            }

        user_request = state.get("user_request")
        if user_request and len(user_request) > 50:  # Complex request
            state["next_action"] = "consult_rag"
        else:
            state["next_action"] = "generate_recommendations"

        state["data_analysis_complete"] = True
        state["messages"] = state["messages"] + [
            AIMessage(content=f"Data analysis complete. Found {len(analysis)} datasets")]
        return state

    def rag_consultant_agent(self, state: AgentState) -> AgentState:
        """RAG consultation agent - gets domain knowledge"""
        user_request = state.get("user_request", "")
        try:
            rag_response = rag.query(user_request)
            state["rag_context"] = str(rag_response)  # Ensure it's a string
            state["messages"] = state["messages"] + [
                AIMessage(content="Retrieved relevant domain knowledge from RAG system")]
        except Exception as e:
            state["rag_context"] = None
            state["messages"] = state["messages"] + [
                AIMessage(content=f"RAG consultation failed: {str(e)}, proceeding without additional context")]

        state["next_action"] = "generate_recommendations"
        return state

    def viz_recommendation_agent(self, state: AgentState) -> AgentState:
        """Visualization recommendation agent"""
        dataframes = state.get("current_dataframes", {})
        user_request = state.get("user_request", "")
        rag_context = state.get("rag_context", "")

        if not dataframes:
            state["error_state"] = "No dataframes available for visualization"
            state["next_action"] = "error"
            return state

        data_summary = {}
        for name, df in dataframes.items():
            data_summary[name] = {
                "columns": list(df.columns)[:10],  # Limit columns
                "sample": df.head(2).to_dict()
            }

        recommendation_prompt = f"""
        You are an expert data visualization consultant. Generate visualization recommendations.

        Available Data: {json.dumps(data_summary, default=str)}
        User Request: {user_request or "Generate insightful visualizations"}
        Domain Context: {rag_context}

        Rules:
        - If user_request is specific: provide 1 targeted recommendation
        - If user_request is general/None: provide 3-5 diverse recommendations
        - Focus on business insights and actionable information

        Return JSON format:
        {{"recommendations": [
            {{
                "title": "Chart Title",
                "plot_type": "bar plot|pie chart|line chart|scatter plot|histogram|heatmap|box plot",
                "columns": ["col1", "col2"], 
                "reasoning": "Why this chart is valuable",
                "data_source": "dataset_name",
                "priority": 1-5
            }}
        ]}}
        """
        try:
            response = self.llm.invoke([HumanMessage(content=recommendation_prompt)])
            recommendations_data = json.loads(response.content)
            state["viz_recommendations"] = recommendations_data.get("recommendations", [])
            state["recommendations_generated"] = True
            state["next_action"] = "create_plots"
            state["messages"] = state["messages"] + [
                AIMessage(content=f"Generated {len(state['viz_recommendations'])} visualization recommendations")]
        except Exception as e:
            state["error_state"] = f"Failed to generate recommendations: {str(e)}"
            state["next_action"] = "error"

        return state

    def plot_generator_agent(self, state: AgentState) -> AgentState:
        """Plot generation agent - creates actual visualizations"""
        recommendations = state.get("viz_recommendations", [])
        dataframes = state.get("current_dataframes", {})

        if not dataframes:
            state["error_state"] = "No data available for plotting"
            state["next_action"] = "error"
            return state

        if not recommendations:
            state["error_state"] = "No recommendations available for plotting"
            state["next_action"] = "error"
            return state

        generated_plots = []

        for rec in recommendations[:5]:  # Limit to 5 plots
            try:
                # Fixed: Changed from 'recommendations.get' to 'rec.get'
                data_source = rec.get("data_source", list(dataframes.keys())[0])
                df = dataframes.get(data_source)

                if df is None:
                    continue

                plot_type = rec.get("plot_type", "bar plot").lower()
                columns = rec.get("columns", [])
                title = rec.get("title", "Visualization")

                # Validate columns exist in dataframe
                valid_columns = [col for col in columns if col in df.columns]
                if not valid_columns and columns:
                    continue

                fig, code = self.create_plot(df, plot_type, valid_columns, title)
                if fig is not None:
                    generated_plots.append({
                        "figure": fig,
                        "code": code,
                        "title": title,
                        "plot_type": plot_type,
                        "reasoning": rec.get("reasoning", ""),
                        "priority": rec.get("priority", 3)
                    })
            except Exception as e:
                print(f"Error generating plot: {e}")
                continue

        state["generated_plots"] = generated_plots
        state["plots_created"] = True
        state["next_action"] = "quality_check"
        state["messages"] = state["messages"] + [AIMessage(content=f"Generated {len(generated_plots)} visualizations")]
        return state

    def quality_checker_agent(self, state: AgentState) -> AgentState:
        """Quality checking agent - validates generated plots"""
        generated_plots = state.get("generated_plots", [])
        user_request = state.get("user_request", "")

        if not generated_plots:
            state["error_state"] = "No plots were generated successfully"
            state["next_action"] = "error"
            return state

        quality_issues = []

        for plot in generated_plots:
            if not plot.get("figure"):
                quality_issues.append("Missing figure object")
            if not plot.get("title"):
                quality_issues.append("Missing plot title")

        if user_request and len(generated_plots) == 0:
            quality_issues.append("No plots match user request")

        if len(quality_issues) > len(generated_plots) / 2:
            state["next_action"] = "regenerate"
            state["messages"] = state["messages"] + [
                AIMessage(content=f"Quality issues found: {quality_issues[:3]}. Regenerating...")]
        else:
            state["next_action"] = "complete"
            state["messages"] = state["messages"] + [
                AIMessage(content="Quality check passed. Visualization generation complete.")]

        return state

    def error_handler_agent(self, state: AgentState) -> AgentState:
        """Error handling agent"""
        error = state.get("error_state", "Unknown error")
        state["messages"] = state["messages"] + [AIMessage(content=f"Error encountered: {error}")]
        return state

    # Routing functions
    def route_from_planner(self, state: AgentState) -> str:
        return state.get("next_action", "end")

    def route_from_data_loader(self, state: AgentState) -> str:
        return state.get("next_action", "error")

    def route_from_analyzer(self, state: AgentState) -> str:
        return state.get("next_action", "error")

    def route_from_recommender(self, state: AgentState) -> str:
        return state.get("next_action", "error")

    def route_from_plot_generator(self, state: AgentState) -> str:
        return state.get("next_action", "error")

    def route_from_quality_checker(self, state: AgentState) -> str:
        return state.get("next_action", "error")

    def create_plot(self, df: pd.DataFrame, plot_type: str, columns: List[str], title: str) -> Tuple[
        Optional[go.Figure], str]:
        """Create plot based on type and parameters"""
        try:
            if plot_type in ["bar plot", "bar chart"] and columns:
                return self._create_bar_chart(df, columns, title)
            elif plot_type == "pie chart" and columns:
                return self._create_pie_chart(df, columns[0], title)
            elif plot_type == "line chart" and len(columns) >= 2:
                return self._create_line_chart(df, columns[0], columns[1], title)
            elif plot_type == "scatter plot" and len(columns) >= 2:
                return self._create_scatter_plot(df, columns[0], columns[1], title)
            elif plot_type == "histogram" and columns:
                return self._create_histogram(df, columns[0], title)
            elif plot_type == "heatmap":
                return self._create_correlation_heatmap(df, title)
            else:
                # Default fallback
                return self._create_bar_chart(df, columns[:1] if columns else [df.columns[0]], title)
        except Exception as e:
            print(f"Plot creation error: {e}")
            return None, ""

    def _create_bar_chart(self, df: pd.DataFrame, columns: List[str], title: str) -> Tuple[go.Figure, str]:
        if not columns:
            return None, ""

        col = columns[0]
        if col not in df.columns:
            return None, ""

        value_counts = df[col].value_counts().head(10)
        fig = px.bar(x=value_counts.index, y=value_counts.values, title=title)
        fig.update_layout(height=500)

        code = f"""
import plotly.express as px
value_counts = df['{col}'].value_counts().head(10)
fig = px.bar(x=value_counts.index, y=value_counts.values, title='{title}')
fig.show()
"""
        return fig, code

    def _create_pie_chart(self, df: pd.DataFrame, column: str, title: str) -> Tuple[go.Figure, str]:
        if column not in df.columns:
            return None, ""

        value_counts = df[column].value_counts().head(8)
        fig = px.pie(values=value_counts.values, names=value_counts.index, title=title)
        fig.update_layout(height=500)

        code = f"""
import plotly.express as px
value_counts = df['{column}'].value_counts().head(8)
fig = px.pie(values=value_counts.values, names=value_counts.index, title='{title}')
fig.show()
"""
        return fig, code

    def _create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Tuple[go.Figure, str]:
        if x_col not in df.columns or y_col not in df.columns:
            return None, ""

        fig = px.line(df, x=x_col, y=y_col, title=title)
        fig.update_layout(height=500)

        code = f"""
import plotly.express as px
fig = px.line(df, x='{x_col}', y='{y_col}', title='{title}')
fig.show()
"""
        return fig, code

    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Tuple[go.Figure, str]:
        if x_col not in df.columns or y_col not in df.columns:
            return None, ""

        fig = px.scatter(df, x=x_col, y=y_col, title=title)
        fig.update_layout(height=500)

        code = f"""
import plotly.express as px
fig = px.scatter(df, x='{x_col}', y='{y_col}', title='{title}')
fig.show()
"""
        return fig, code

    def _create_histogram(self, df: pd.DataFrame, column: str, title: str) -> Tuple[go.Figure, str]:
        if column not in df.columns:
            return None, ""

        fig = px.histogram(df, x=column, title=title)
        fig.update_layout(height=500)

        code = f"""
import plotly.express as px
fig = px.histogram(df, x='{column}', title='{title}')
fig.show()
"""
        return fig, code

    def _create_correlation_heatmap(self, df: pd.DataFrame, title: str) -> Tuple[go.Figure, str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return None, ""

        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, title=title)
        fig.update_layout(height=500)

        code = f"""
import plotly.express as px
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
corr_matrix = df[numeric_cols].corr()
fig = px.imshow(corr_matrix, text_auto=True, title='{title}')
fig.show()
"""
        return fig, code

    def execute_visualization_task(self, file_names: List[str], user_request: str = None) -> Dict[str, Any]:
        """Execute the complete agentic visualization workflow"""
        initial_state = AgentState(
            messages=[HumanMessage(content=user_request or "Generate insightful visualizations")],
            file_names=file_names,
            user_request=user_request,
            data_loaded=False,
            data_analysis_complete=False,
            recommendations_generated=False,
            plots_created=False,
            current_dataframes={},
            viz_recommendations=[],
            generated_plots=[],
            error_state=None,
            next_action="start",
            rag_context=None
        )

        try:
            final_state = self.graph.invoke(initial_state)
            return {
                "success": final_state.get("error_state") is None,
                "plots": final_state.get("generated_plots", []),
                "messages": [msg.content for msg in final_state.get("messages", [])],
                "error": final_state.get("error_state"),
                "recommendations": final_state.get("viz_recommendations", [])
            }
        except Exception as e:
            return {
                "success": False,
                "plots": [],
                "messages": [f"Workflow execution failed: {str(e)}"],
                "error": str(e),
                "recommendations": []
            }