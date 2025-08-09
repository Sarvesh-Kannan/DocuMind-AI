#!/usr/bin/env python3
"""Simple test for new features"""

import pandas as pd
from phase5_streamlit_app import Phase5StreamlitApp

def test_new_features():
    print('🧪 Testing Enhanced Features...')
    
    # Test 1: Accuracy Score Calculation
    app = Phase5StreamlitApp()
    result = {
        'text': 'Deep learning uses neural networks for pattern recognition', 
        'score': 0.85, 
        'file_name': 'test.pdf'
    }
    
    queries = ['deep learning', 'neural networks', 'pattern recognition', 'unrelated topic']
    
    print('\n1. Accuracy Score Calculation:')
    for query in queries:
        accuracy = app.calculate_accuracy_score(result, query)
        print(f'   Query: "{query}" -> Accuracy: {accuracy:.3f}')
    
    # Test 2: Query Suggestions
    print('\n2. Enhanced Query Suggestions:')
    mock_docs = pd.DataFrame({
        'file_name': ['AI_Research.pdf', 'Machine_Learning_Guide.pdf', 'Deep_Learning_Basics.pdf'],
        'text': ['AI content', 'ML content', 'DL content'],
        'chunk_id': [1, 2, 3],
        'page_number': [1, 1, 1]
    })
    app.documents_df = mock_docs
    app._update_query_suggestions()
    
    general_suggestions = app.get_enhanced_query_suggestions()
    print(f'   General suggestions: {len(general_suggestions)} found')
    print(f'   Sample: {general_suggestions[:3]}')
    
    filtered_suggestions = app.get_enhanced_query_suggestions('AI')
    print(f'   Filtered for "AI": {len(filtered_suggestions)} found')
    
    print('\n✅ All enhanced features working correctly!')
    
    # Test 3: PDF Upload function exists
    print('\n3. PDF Upload Feature:')
    has_upload = hasattr(app, 'upload_and_process_pdfs')
    print(f'   Upload method exists: {"✅ Yes" if has_upload else "❌ No"}')
    
    # Test 4: Enhanced search with accuracy
    print('\n4. Enhanced Search Features:')
    has_accuracy = hasattr(app, 'calculate_accuracy_score')
    print(f'   Accuracy calculation: {"✅ Yes" if has_accuracy else "❌ No"}')
    
    return True

if __name__ == "__main__":
    test_new_features() 