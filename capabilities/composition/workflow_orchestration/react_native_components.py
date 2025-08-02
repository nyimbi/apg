"""
React Native Component Generator

Generates React Native components for mobile workflow applications:
- Native iOS/Android components
- Cross-platform workflow components
- Mobile-optimized UI elements
- Touch gesture handlers
- Offline capability support
- Push notification integration

© 2025 Datacraft
Author: Nyimbi Odero
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ReactNativeComponent:
	"""React Native component definition"""
	name: str
	component_type: str  # screen, component, hook, service
	props: Dict[str, Any]
	dependencies: List[str]
	platform_specific: Dict[str, Any]  # ios, android specific code
	accessibility: Dict[str, Any]
	performance_hints: List[str]


class ReactNativeGenerator:
	"""Generates React Native components for mobile workflows"""
	
	def __init__(self):
		self.components: Dict[str, ReactNativeComponent] = {}
		self.generated_files: Dict[str, str] = {}
	
	def generate_workflow_screen(self, workflow_data: Dict[str, Any]) -> str:
		"""Generate React Native screen for workflow"""
		try:
			workflow_id = workflow_data.get('id', 'unknown')
			workflow_name = workflow_data.get('name', 'Workflow')
			
			screen_code = f'''
import React, {{ useState, useEffect, useCallback }} from 'react';
import {{
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  Animated,
  Dimensions,
  Platform
}} from 'react-native';
import {{ SafeAreaView }} from 'react-native-safe-area-context';
import {{ useFocusEffect, useNavigation }} from '@react-navigation/native';
import {{ useSelector, useDispatch }} from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialIcons';
import LinearGradient from 'react-native-linear-gradient';
import Haptics from 'react-native-haptic-feedback';
import NetInfo from '@react-native-async-storage/async-storage';

// Custom hooks
import {{ useWorkflow }} from '../hooks/useWorkflow';
import {{ useOfflineSync }} from '../hooks/useOfflineSync';
import {{ usePushNotifications }} from '../hooks/usePushNotifications';

// Components
import WorkflowStepCard from '../components/WorkflowStepCard';
import ProgressIndicator from '../components/ProgressIndicator';
import OfflineBanner from '../components/OfflineBanner';
import AccessibilityWrapper from '../components/AccessibilityWrapper';

const {{ width, height }} = Dimensions.get('window');

const {workflow_name.replace(' ', '')}Screen = ({{ route }}) => {{
  const navigation = useNavigation();
  const dispatch = useDispatch();
  
  // Workflow state
  const {{
    workflow,
    execution,
    currentStep,
    progress,
    isLoading,
    error,
    executeWorkflow,
    cancelExecution,
    nextStep,
    previousStep
  }} = useWorkflow('{workflow_id}');
  
  // Offline capabilities
  const {{
    isOnline,
    syncStatus,
    pendingSync,
    syncData
  }} = useOfflineSync();
  
  // Push notifications
  const {{ requestPermissions, scheduleNotification }} = usePushNotifications();
  
  // Local state
  const [isExecuting, setIsExecuting] = useState(false);
  const [expandedStep, setExpandedStep] = useState(null);
  const fadeAnim = useState(new Animated.Value(0))[0];
  
  // Animation
  useEffect(() => {{
    Animated.timing(fadeAnim, {{
      toValue: 1,
      duration: 300,
      useNativeDriver: true
    }}).start();
  }}, []);
  
  // Focus effect for screen updates
  useFocusEffect(
    useCallback(() => {{
      // Refresh data when screen comes into focus
      if (execution?.id) {{
        // Poll for execution status
        const interval = setInterval(() => {{
          // Update execution status
        }}, 5000);
        
        return () => clearInterval(interval);
      }}
    }}, [execution?.id])
  );
  
  // Handle workflow execution
  const handleExecute = useCallback(async () => {{
    try {{
      setIsExecuting(true);
      
      // Haptic feedback
      Haptics.impact(Haptics.ImpactFeedbackStyle.Medium);
      
      // Request notification permissions
      await requestPermissions();
      
      // Execute workflow
      const result = await executeWorkflow({{
        parameters: route.params?.parameters || {{}}
      }});
      
      if (result.success) {{
        // Schedule completion notification
        await scheduleNotification({{
          title: 'Workflow Started',
          message: `{workflow_name} has been started`,
          data: {{ executionId: result.executionId }}
        }});
        
        // Navigate to execution screen
        navigation.navigate('WorkflowExecution', {{
          executionId: result.executionId
        }});
      }} else {{
        Alert.alert('Error', result.error || 'Failed to start workflow');
      }}
    }} catch (err) {{
      Alert.alert('Error', err.message || 'Failed to execute workflow');
    }} finally {{
      setIsExecuting(false);
    }}
  }}, [executeWorkflow, navigation, requestPermissions, scheduleNotification]);
  
  // Handle step expansion
  const handleStepPress = useCallback((stepId) => {{
    Haptics.selection();
    setExpandedStep(expandedStep === stepId ? null : stepId);
  }}, [expandedStep]);
  
  // Render workflow step
  const renderStep = useCallback((step, index) => (
    <WorkflowStepCard
      key={{step.id}}
      step={{step}}
      index={{index}}
      isExpanded={{expandedStep === step.id}}
      onPress={{() => handleStepPress(step.id)}}
      isOfflineCapable={{step.offline_capable}}
      estimatedTime={{step.estimated_time}}
      accessibility={{{{
        label: `Step ${{index + 1}}: ${{step.name}}`,
        hint: step.description,
        role: 'button'
      }}}}
    />
  ), [expandedStep, handleStepPress]);
  
  // Loading state
  if (isLoading) {{
    return (
      <SafeAreaView style={{styles.container}}>
        <View style={{styles.loadingContainer}}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={{styles.loadingText}}>Loading workflow...</Text>
        </View>
      </SafeAreaView>
    );
  }}
  
  // Error state
  if (error) {{
    return (
      <SafeAreaView style={{styles.container}}>
        <View style={{styles.errorContainer}}>
          <Icon name="error" size={{48}} color="#FF3B30" />
          <Text style={{styles.errorTitle}}>Error Loading Workflow</Text>
          <Text style={{styles.errorMessage}}>{{error}}</Text>
          <TouchableOpacity
            style={{styles.retryButton}}
            onPress={{() => window.location.reload()}}
            accessibilityLabel="Retry loading workflow"
          >
            <Text style={{styles.retryButtonText}}>Retry</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }}
  
  return (
    <SafeAreaView style={{styles.container}}>
      {{/* Offline Banner */}}
      {{!isOnline && <OfflineBanner syncStatus={{syncStatus}} onSync={{syncData}} />}}
      
      <Animated.View style={{[styles.content, {{ opacity: fadeAnim }}]}}>
        {{/* Header */}}
        <LinearGradient
          colors={{['#007AFF', '#5856D6']}}
          style={{styles.header}}
        >
          <View style={{styles.headerContent}}>
            <TouchableOpacity
              style={{styles.backButton}}
              onPress={{() => navigation.goBack()}}
              accessibilityLabel="Go back"
            >
              <Icon name="arrow-back" size={{24}} color="white" />
            </TouchableOpacity>
            
            <View style={{styles.headerText}}>
              <Text style={{styles.headerTitle}}>{workflow_name}</Text>
              <Text style={{styles.headerSubtitle}}>
                {{workflow?.node_count || 0}} steps • {{workflow?.estimated_time || '5-10'}} min
              </Text>
            </View>
            
            <TouchableOpacity
              style={{styles.moreButton}}
              onPress={{() => {{/* Show workflow options */}}}}
              accessibilityLabel="More options"
            >
              <Icon name="more-vert" size={{24}} color="white" />
            </TouchableOpacity>
          </View>
        </LinearGradient>
        
        {{/* Content */}}
        <ScrollView
          style={{styles.scrollView}}
          contentContainerStyle={{styles.scrollContent}}
          showsVerticalScrollIndicator={{false}}
        >
          {{/* Workflow Description */}}
          <View style={{styles.descriptionCard}}>
            <Text style={{styles.descriptionTitle}}>About this workflow</Text>
            <Text style={{styles.descriptionText}}>
              {{workflow?.description || 'No description available'}}
            </Text>
            
            {{/* Workflow Metadata */}}
            <View style={{styles.metadataContainer}}>
              <View style={{styles.metadataItem}}>
                <Icon name="schedule" size={{16}} color="#8E8E93" />
                <Text style={{styles.metadataText}}>
                  {{workflow?.estimated_time || '5-10'}} minutes
                </Text>
              </View>
              
              <View style={{styles.metadataItem}}>
                <Icon name="trending-up" size={{16}} color="#8E8E93" />
                <Text style={{styles.metadataText}}>
                  {{workflow?.complexity || 'Medium'}} complexity
                </Text>
              </View>
              
              {{workflow?.offline_capable && (
                <View style={{styles.metadataItem}}>
                  <Icon name="cloud-off" size={{16}} color="#34C759" />
                  <Text style={{[styles.metadataText, {{ color: '#34C759' }}]}}>
                    Offline capable
                  </Text>
                </View>
              )}}
            </View>
          </View>
          
          {{/* Workflow Steps */}}
          <View style={{styles.stepsContainer}}>
            <Text style={{styles.stepsTitle}}>Workflow Steps</Text>
            {{workflow?.mobile_steps?.map(renderStep)}}
          </View>
          
          {{/* Prerequisites */}}
          {{workflow?.prerequisites?.length > 0 && (
            <View style={{styles.prerequisitesCard}}>
              <Text style={{styles.prerequisitesTitle}}>Prerequisites</Text>
              {{workflow.prerequisites.map((prereq, index) => (
                <View key={{index}} style={{styles.prerequisiteItem}}>
                  <Icon name="check-circle" size={{16}} color="#34C759" />
                  <Text style={{styles.prerequisiteText}}>{{prereq}}</Text>
                </View>
              ))}}
            </View>
          )}}
        </ScrollView>
        
        {{/* Execute Button */}}
        <View style={{styles.bottomContainer}}>
          {{workflow?.requires_approval && (
            <View style={{styles.warningContainer}}>
              <Icon name="warning" size={{16}} color="#FF9500" />
              <Text style={{styles.warningText}}>
                This workflow requires approval before execution
              </Text>
            </View>
          )}}
          
          <TouchableOpacity
            style={{[
              styles.executeButton,
              isExecuting && styles.executeButtonDisabled
            ]}}
            onPress={{handleExecute}}
            disabled={{isExecuting}}
            accessibilityLabel="Execute workflow"
            accessibilityHint="Starts the workflow execution"
          >
            {{isExecuting ? (
              <ActivityIndicator size="small" color="white" />
            ) : (
              <Icon name="play-arrow" size={{24}} color="white" />
            )}}
            <Text style={{styles.executeButtonText}}>
              {{isExecuting ? 'Starting...' : 'Execute Workflow'}}
            </Text>
          </TouchableOpacity>
        </View>
      </Animated.View>
    </SafeAreaView>
  );
}};

const styles = StyleSheet.create({{
  container: {{
    flex: 1,
    backgroundColor: '#F2F2F7'
  }},
  content: {{
    flex: 1
  }},
  loadingContainer: {{
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  }},
  loadingText: {{
    marginTop: 16,
    fontSize: 16,
    color: '#8E8E93',
    fontWeight: '500'
  }},
  errorContainer: {{
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20
  }},
  errorTitle: {{
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FF3B30',
    marginTop: 16,
    marginBottom: 8
  }},
  errorMessage: {{
    fontSize: 16,
    color: '#8E8E93',
    textAlign: 'center',
    marginBottom: 24
  }},
  retryButton: {{
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8
  }},
  retryButtonText: {{
    color: 'white',
    fontSize: 16,
    fontWeight: '600'
  }},
  header: {{
    paddingTop: Platform.OS === 'ios' ? 0 : 20
  }},
  headerContent: {{
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    paddingBottom: 20
  }},
  backButton: {{
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.2)'
  }},
  headerText: {{
    flex: 1,
    marginLeft: 16
  }},
  headerTitle: {{
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white'
  }},
  headerSubtitle: {{
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 2
  }},
  moreButton: {{
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.2)'
  }},
  scrollView: {{
    flex: 1
  }},
  scrollContent: {{
    padding: 16,
    paddingBottom: 120
  }},
  descriptionCard: {{
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: {{ width: 0, height: 1 }},
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2
  }},
  descriptionTitle: {{
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1C1C1E',
    marginBottom: 8
  }},
  descriptionText: {{
    fontSize: 16,
    color: '#8E8E93',
    lineHeight: 24,
    marginBottom: 16
  }},
  metadataContainer: {{
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12
  }},
  metadataItem: {{
    flexDirection: 'row',
    alignItems: 'center'
  }},
  metadataText: {{
    fontSize: 14,
    color: '#8E8E93',
    marginLeft: 4
  }},
  stepsContainer: {{
    marginBottom: 16
  }},
  stepsTitle: {{
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1C1C1E',
    marginBottom: 12
  }},
  prerequisitesCard: {{
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: {{ width: 0, height: 1 }},
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2
  }},
  prerequisitesTitle: {{
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1C1C1E',
    marginBottom: 12
  }},
  prerequisiteItem: {{
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8
  }},
  prerequisiteText: {{
    fontSize: 14,
    color: '#3C3C43',
    marginLeft: 8,
    flex: 1
  }},
  bottomContainer: {{
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'white',
    padding: 16,
    paddingBottom: Platform.OS === 'ios' ? 34 : 16,
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA'
  }},
  warningContainer: {{
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF3CD',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12
  }},
  warningText: {{
    fontSize: 14,
    color: '#856404',
    marginLeft: 8,
    flex: 1
  }},
  executeButton: {{
    backgroundColor: '#007AFF',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#007AFF',
    shadowOffset: {{ width: 0, height: 4 }},
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4
  }},
  executeButtonDisabled: {{
    backgroundColor: '#8E8E93',
    shadowOpacity: 0
  }},
  executeButtonText: {{
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8
  }}
}});

export default {workflow_name.replace(' ', '')}Screen;
'''
			
			self.generated_files[f"{workflow_name.replace(' ', '')}Screen.jsx"] = screen_code
			return screen_code
			
		except Exception as e:
			logger.error(f"Generate workflow screen error: {e}")
			raise
	
	def generate_workflow_step_component(self) -> str:
		"""Generate WorkflowStepCard component"""
		try:
			component_code = '''
import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  LayoutAnimation,
  Platform
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import Haptics from 'react-native-haptic-feedback';

const WorkflowStepCard = ({
  step,
  index,
  isExpanded,
  onPress,
  isOfflineCapable,
  estimatedTime,
  accessibility
}) => {
  const rotateAnim = useRef(new Animated.Value(0)).current;
  
  React.useEffect(() => {
    Animated.timing(rotateAnim, {
      toValue: isExpanded ? 1 : 0,
      duration: 200,
      useNativeDriver: true
    }).start();
  }, [isExpanded, rotateAnim]);
  
  const handlePress = () => {
    if (Platform.OS === 'ios') {
      LayoutAnimation.easeInEaseOut();
    }
    Haptics.selection();
    onPress();
  };
  
  const getStepIcon = (stepType) => {
    switch (stepType) {
      case 'input': return 'input';
      case 'processing': return 'settings';
      case 'decision': return 'decision';
      case 'output': return 'output';
      case 'approval': return 'how-to-vote';
      default: return 'radio-button-unchecked';
    }
  };
  
  const getStepColor = (stepType) => {
    switch (stepType) {
      case 'input': return '#007AFF';
      case 'processing': return '#5856D6';
      case 'decision': return '#FF9500';
      case 'output': return '#34C759';
      case 'approval': return '#FF3B30';
      default: return '#8E8E93';
    }
  };
  
  const rotate = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '180deg']
  });
  
  return (
    <TouchableOpacity
      style={styles.container}
      onPress={handlePress}
      activeOpacity={0.7}
      accessible={true}
      accessibilityLabel={accessibility?.label}
      accessibilityHint={accessibility?.hint}
      accessibilityRole={accessibility?.role || 'button'}
    >
      <View style={styles.card}>
        {/* Step Header */}
        <View style={styles.header}>
          <View style={styles.leftContent}>
            {/* Step Number */}
            <View style={[styles.stepNumber, { backgroundColor: getStepColor(step.type) }]}>
              <Text style={styles.stepNumberText}>{index + 1}</Text>
            </View>
            
            {/* Step Info */}
            <View style={styles.stepInfo}>
              <Text style={styles.stepTitle}>{step.name}</Text>
              <View style={styles.stepMeta}>
                <Icon 
                  name={getStepIcon(step.type)} 
                  size={14} 
                  color={getStepColor(step.type)} 
                />
                <Text style={styles.stepType}>{step.type}</Text>
                {estimatedTime && (
                  <>
                    <Text style={styles.separator}>•</Text>
                    <Icon name="schedule" size={14} color="#8E8E93" />
                    <Text style={styles.stepDuration}>{estimatedTime}s</Text>
                  </>
                )}
              </View>
            </View>
          </View>
          
          {/* Right Content */}
          <View style={styles.rightContent}>
            {isOfflineCapable && (
              <View style={styles.offlineBadge}>
                <Icon name="cloud-off" size={12} color="#34C759" />
              </View>
            )}
            
            <Animated.View style={{ transform: [{ rotate }] }}>
              <Icon 
                name="keyboard-arrow-down" 
                size={24} 
                color="#8E8E93" 
              />
            </Animated.View>
          </View>
        </View>
        
        {/* Expanded Content */}
        {isExpanded && (
          <View style={styles.expandedContent}>
            <Text style={styles.description}>{step.description}</Text>
            
            {step.required_input?.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Required Input</Text>
                {step.required_input.map((input, idx) => (
                  <View key={idx} style={styles.inputItem}>
                    <Icon name="input" size={16} color="#007AFF" />
                    <Text style={styles.inputText}>{input}</Text>
                  </View>
                ))}
              </View>
            )}
            
            {step.touch_interactions?.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Interactions</Text>
                <View style={styles.interactionTags}>
                  {step.touch_interactions.map((interaction, idx) => (
                    <View key={idx} style={styles.interactionTag}>
                      <Text style={styles.interactionText}>
                        {interaction.replace('_', ' ')}
                      </Text>
                    </View>
                  ))}
                </View>
              </View>
            )}
          </View>
        )}
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 12
  },
  card: {
    backgroundColor: 'white',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16
  },
  leftContent: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center'
  },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12
  },
  stepNumberText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold'
  },
  stepInfo: {
    flex: 1
  },
  stepTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1C1C1E',
    marginBottom: 4
  },
  stepMeta: {
    flexDirection: 'row',
    alignItems: 'center'
  },
  stepType: {
    fontSize: 12,
    color: '#8E8E93',
    marginLeft: 4,
    textTransform: 'capitalize'
  },
  separator: {
    fontSize: 12,
    color: '#8E8E93',
    marginHorizontal: 6
  },
  stepDuration: {
    fontSize: 12,
    color: '#8E8E93',
    marginLeft: 4
  },
  rightContent: {
    flexDirection: 'row',
    alignItems: 'center'
  },
  offlineBadge: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#E8F5E8',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8
  },
  expandedContent: {
    padding: 16,
    paddingTop: 0,
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA'
  },
  description: {
    fontSize: 14,
    color: '#3C3C43',
    lineHeight: 20,
    marginBottom: 16
  },
  section: {
    marginBottom: 16
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1C1C1E',
    marginBottom: 8
  },
  inputItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6
  },
  inputText: {
    fontSize: 14,
    color: '#3C3C43',
    marginLeft: 8
  },
  interactionTags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8
  },
  interactionTag: {
    backgroundColor: '#F2F2F7',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6
  },
  interactionText: {
    fontSize: 12,
    color: '#3C3C43',
    textTransform: 'capitalize'
  }
});

export default WorkflowStepCard;
'''
			
			self.generated_files["WorkflowStepCard.jsx"] = component_code
			return component_code
			
		except Exception as e:
			logger.error(f"Generate step component error: {e}")
			raise
	
	def generate_workflow_hooks(self) -> str:
		"""Generate React Native hooks for workflow management"""
		try:
			hooks_code = '''
import { useState, useEffect, useCallback, useRef } from 'react';
import { Alert, AppState } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import BackgroundJob from 'react-native-background-job';
import PushNotification from 'react-native-push-notification';

// API Service
import * as WorkflowAPI from '../services/WorkflowAPI';
import * as OfflineStorage from '../services/OfflineStorage';

/**
 * Hook for managing workflow operations
 */
export const useWorkflow = (workflowId) => {
  const [workflow, setWorkflow] = useState(null);
  const [execution, setExecution] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const intervalRef = useRef(null);
  
  // Load workflow details
  useEffect(() => {
    const loadWorkflow = async () => {
      try {
        setIsLoading(true);
        const workflowData = await WorkflowAPI.getWorkflowDetails(workflowId);
        setWorkflow(workflowData);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    
    if (workflowId) {
      loadWorkflow();
    }
  }, [workflowId]);
  
  // Execute workflow
  const executeWorkflow = useCallback(async (params) => {
    try {
      const result = await WorkflowAPI.executeWorkflow(workflowId, params);
      if (result.success) {
        setExecution(result.data);
        
        // Start polling for updates
        intervalRef.current = setInterval(async () => {
          try {
            const status = await WorkflowAPI.getExecutionStatus(result.data.execution_id);
            setExecution(status.data);
            setProgress(status.data.progress);
            setCurrentStep(status.data.current_step?.step_number || 0);
            
            // Stop polling if execution is complete
            if (['completed', 'failed', 'cancelled'].includes(status.data.status)) {
              clearInterval(intervalRef.current);
            }
          } catch (err) {
            console.warn('Failed to update execution status:', err);
          }
        }, 5000);
        
        return result;
      } else {
        throw new Error(result.error);
      }
    } catch (err) {
      setError(err.message);
      return { success: false, error: err.message };
    }
  }, [workflowId]);
  
  // Cancel execution
  const cancelExecution = useCallback(async () => {
    try {
      if (execution?.execution_id) {
        await WorkflowAPI.cancelExecution(execution.execution_id);
        clearInterval(intervalRef.current);
        setExecution({ ...execution, status: 'cancelled' });
      }
    } catch (err) {
      Alert.alert('Error', 'Failed to cancel workflow execution');
    }
  }, [execution]);
  
  // Navigation helpers
  const nextStep = useCallback(() => {
    if (currentStep < (workflow?.mobile_steps?.length || 0) - 1) {
      setCurrentStep(currentStep + 1);
    }
  }, [currentStep, workflow]);
  
  const previousStep = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);
  
  // Cleanup
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);
  
  return {
    workflow,
    execution,
    currentStep,
    progress,
    isLoading,
    error,
    executeWorkflow,
    cancelExecution,
    nextStep,
    previousStep
  };
};

/**
 * Hook for offline synchronization
 */
export const useOfflineSync = () => {
  const [isOnline, setIsOnline] = useState(true);
  const [syncStatus, setSyncStatus] = useState('idle');
  const [pendingSync, setPendingSync] = useState([]);
  const [lastSyncTime, setLastSyncTime] = useState(null);
  
  // Monitor network connectivity
  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener(state => {
      setIsOnline(state.isConnected);
      
      // Auto-sync when coming back online
      if (state.isConnected && pendingSync.length > 0) {
        syncData();
      }
    });
    
    return unsubscribe;
  }, [pendingSync]);
  
  // Load pending sync items
  useEffect(() => {
    const loadPendingSync = async () => {
      try {
        const pending = await OfflineStorage.getPendingSyncItems();
        setPendingSync(pending);
        
        const lastSync = await AsyncStorage.getItem('lastSyncTime');
        if (lastSync) {
          setLastSyncTime(new Date(lastSync));
        }
      } catch (err) {
        console.warn('Failed to load pending sync items:', err);
      }
    };
    
    loadPendingSync();
  }, []);
  
  // Sync data with server
  const syncData = useCallback(async () => {
    if (!isOnline) {
      Alert.alert('Offline', 'Cannot sync while offline');
      return;
    }
    
    try {
      setSyncStatus('syncing');
      
      // Upload pending changes
      for (const item of pendingSync) {
        await WorkflowAPI.uploadPendingChange(item);
      }
      
      // Clear pending items
      await OfflineStorage.clearPendingSyncItems();
      setPendingSync([]);
      
      // Update last sync time
      const now = new Date();
      await AsyncStorage.setItem('lastSyncTime', now.toISOString());
      setLastSyncTime(now);
      
      setSyncStatus('completed');
      setTimeout(() => setSyncStatus('idle'), 2000);
      
    } catch (err) {
      setSyncStatus('error');
      Alert.alert('Sync Error', err.message);
    }
  }, [isOnline, pendingSync]);
  
  // Add item to pending sync
  const addToPendingSync = useCallback(async (item) => {
    try {
      await OfflineStorage.addPendingSyncItem(item);
      setPendingSync(prev => [...prev, item]);
    } catch (err) {
      console.warn('Failed to add pending sync item:', err);
    }
  }, []);
  
  return {
    isOnline,
    syncStatus,
    pendingSync,
    lastSyncTime,
    syncData,
    addToPendingSync
  };
};

/**
 * Hook for push notifications
 */
export const usePushNotifications = () => {
  const [hasPermission, setHasPermission] = useState(false);
  const [notifications, setNotifications] = useState([]);
  
  // Request permissions
  const requestPermissions = useCallback(async () => {
    try {
      PushNotification.requestPermissions().then(permissions => {
        setHasPermission(permissions.alert && permissions.badge && permissions.sound);
      });
    } catch (err) {
      console.warn('Failed to request notification permissions:', err);
    }
  }, []);
  
  // Configure notifications
  useEffect(() => {
    PushNotification.configure({
      onRegister: function(token) {
        // Send token to server
        WorkflowAPI.updateDeviceToken(token.token);
      },
      
      onNotification: function(notification) {
        if (notification.userInteraction) {
          // Handle notification tap
          handleNotificationTap(notification);
        }
        
        // Add to local notifications list
        setNotifications(prev => [notification, ...prev.slice(0, 49)]); // Keep last 50
      },
      
      permissions: {
        alert: true,
        badge: true,
        sound: true
      },
      
      popInitialNotification: true,
      requestPermissions: false
    });
    
    // Request permissions on mount
    requestPermissions();
  }, [requestPermissions]);
  
  // Handle notification tap
  const handleNotificationTap = useCallback((notification) => {
    const { data } = notification;
    
    if (data?.executionId) {
      // Navigate to execution screen
      // This would use your navigation system
      console.log('Navigate to execution:', data.executionId);
    } else if (data?.workflowId) {
      // Navigate to workflow screen
      console.log('Navigate to workflow:', data.workflowId);
    }
  }, []);
  
  // Schedule local notification
  const scheduleNotification = useCallback(async (options) => {
    if (!hasPermission) {
      console.warn('No notification permission');
      return;
    }
    
    try {
      PushNotification.localNotification({
        title: options.title,
        message: options.message,
        userInfo: options.data || {},
        soundName: 'default',
        playSound: true,
        vibrate: true,
        vibration: 300
      });
    } catch (err) {
      console.warn('Failed to schedule notification:', err);
    }
  }, [hasPermission]);
  
  // Clear notifications
  const clearNotifications = useCallback(() => {
    PushNotification.removeAllDeliveredNotifications();
    setNotifications([]);
  }, []);
  
  return {
    hasPermission,
    notifications,
    requestPermissions,
    scheduleNotification,
    clearNotifications
  };
};

/**
 * Hook for app state management
 */
export const useAppState = () => {
  const [appState, setAppState] = useState(AppState.currentState);
  const [isActive, setIsActive] = useState(true);
  
  useEffect(() => {
    const handleAppStateChange = (nextAppState) => {
      setAppState(nextAppState);
      setIsActive(nextAppState === 'active');
      
      // Handle background/foreground transitions
      if (nextAppState === 'background') {
        // App went to background
        BackgroundJob.start({
          jobKey: 'workflowSync',
          period: 15000 // 15 seconds
        });
      } else if (nextAppState === 'active') {
        // App came to foreground
        BackgroundJob.stop();
      }
    };
    
    const subscription = AppState.addEventListener('change', handleAppStateChange);
    
    return () => subscription?.remove();
  }, []);
  
  return {
    appState,
    isActive
  };
};
'''
			
			self.generated_files["useWorkflow.js"] = hooks_code
			return hooks_code
			
		except Exception as e:
			logger.error(f"Generate hooks error: {e}")
			raise
	
	def generate_offline_storage_service(self) -> str:
		"""Generate offline storage service"""
		try:
			storage_code = '''
import AsyncStorage from '@react-native-async-storage/async-storage';
import RNFS from 'react-native-fs';
import { zip, unzip } from 'react-native-zip-archive';

const STORAGE_KEYS = {
  WORKFLOWS: 'cached_workflows',
  EXECUTIONS: 'cached_executions',
  PENDING_SYNC: 'pending_sync_items',
  USER_PREFERENCES: 'user_preferences',
  OFFLINE_ASSETS: 'offline_assets'
};

const MAX_CACHE_SIZE = 100 * 1024 * 1024; // 100MB

class OfflineStorageService {
  constructor() {
    this.initializeStorage();
  }
  
  async initializeStorage() {
    try {
      // Create offline directory
      const offlineDir = `${RNFS.DocumentDirectoryPath}/offline`;
      if (!(await RNFS.exists(offlineDir))) {
        await RNFS.mkdir(offlineDir);
      }
      
      // Check cache size
      await this.cleanupOldCache();
    } catch (err) {
      console.warn('Failed to initialize offline storage:', err);
    }
  }
  
  // Workflow caching
  async cacheWorkflow(workflowId, workflowData) {
    try {
      const cached = await this.getCachedWorkflows();
      cached[workflowId] = {
        ...workflowData,
        cachedAt: new Date().toISOString(),
        version: workflowData.version || '1.0'
      };
      
      await AsyncStorage.setItem(STORAGE_KEYS.WORKFLOWS, JSON.stringify(cached));
      
      // Cache associated assets
      if (workflowData.assets) {
        await this.cacheAssets(workflowId, workflowData.assets);
      }
      
    } catch (err) {
      console.warn('Failed to cache workflow:', err);
    }
  }
  
  async getCachedWorkflow(workflowId) {
    try {
      const cached = await this.getCachedWorkflows();
      return cached[workflowId] || null;
    } catch (err) {
      console.warn('Failed to get cached workflow:', err);
      return null;
    }
  }
  
  async getCachedWorkflows() {
    try {
      const cached = await AsyncStorage.getItem(STORAGE_KEYS.WORKFLOWS);
      return cached ? JSON.parse(cached) : {};
    } catch (err) {
      console.warn('Failed to get cached workflows:', err);
      return {};
    }
  }
  
  async removeCachedWorkflow(workflowId) {
    try {
      const cached = await this.getCachedWorkflows();
      delete cached[workflowId];
      await AsyncStorage.setItem(STORAGE_KEYS.WORKFLOWS, JSON.stringify(cached));
      
      // Remove associated assets
      await this.removeAssets(workflowId);
    } catch (err) {
      console.warn('Failed to remove cached workflow:', err);
    }
  }
  
  // Execution caching
  async cacheExecution(executionId, executionData) {
    try {
      const cached = await this.getCachedExecutions();
      cached[executionId] = {
        ...executionData,
        cachedAt: new Date().toISOString()
      };
      
      await AsyncStorage.setItem(STORAGE_KEYS.EXECUTIONS, JSON.stringify(cached));
    } catch (err) {
      console.warn('Failed to cache execution:', err);
    }
  }
  
  async getCachedExecution(executionId) {
    try {
      const cached = await this.getCachedExecutions();
      return cached[executionId] || null;
    } catch (err) {
      console.warn('Failed to get cached execution:', err);
      return null;
    }
  }
  
  async getCachedExecutions() {
    try {
      const cached = await AsyncStorage.getItem(STORAGE_KEYS.EXECUTIONS);
      return cached ? JSON.parse(cached) : {};
    } catch (err) {
      console.warn('Failed to get cached executions:', err);
      return {};
    }
  }
  
  // Pending sync management
  async addPendingSyncItem(item) {
    try {
      const pending = await this.getPendingSyncItems();
      const syncItem = {
        id: `sync_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        ...item,
        createdAt: new Date().toISOString()
      };
      
      pending.push(syncItem);
      await AsyncStorage.setItem(STORAGE_KEYS.PENDING_SYNC, JSON.stringify(pending));
      
      return syncItem.id;
    } catch (err) {
      console.warn('Failed to add pending sync item:', err);
      throw err;
    }
  }
  
  async getPendingSyncItems() {
    try {
      const pending = await AsyncStorage.getItem(STORAGE_KEYS.PENDING_SYNC);
      return pending ? JSON.parse(pending) : [];
    } catch (err) {
      console.warn('Failed to get pending sync items:', err);
      return [];
    }
  }
  
  async removePendingSyncItem(itemId) {
    try {
      const pending = await this.getPendingSyncItems();
      const filtered = pending.filter(item => item.id !== itemId);
      await AsyncStorage.setItem(STORAGE_KEYS.PENDING_SYNC, JSON.stringify(filtered));
    } catch (err) {
      console.warn('Failed to remove pending sync item:', err);
    }
  }
  
  async clearPendingSyncItems() {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.PENDING_SYNC, JSON.stringify([]));
    } catch (err) {
      console.warn('Failed to clear pending sync items:', err);
    }
  }
  
  // Asset caching
  async cacheAssets(workflowId, assets) {
    try {
      const workflowDir = `${RNFS.DocumentDirectoryPath}/offline/${workflowId}`;
      if (!(await RNFS.exists(workflowDir))) {
        await RNFS.mkdir(workflowDir);
      }
      
      for (const asset of assets) {
        if (asset.url && asset.type) {
          const fileName = `${asset.id || asset.name}.${asset.type}`;
          const filePath = `${workflowDir}/${fileName}`;
          
          // Download and cache asset
          await RNFS.downloadFile({
            fromUrl: asset.url,
            toFile: filePath
          }).promise;
          
          // Compress if it's a large file
          const stats = await RNFS.stat(filePath);
          if (stats.size > 1024 * 1024) { // 1MB
            const compressedPath = `${filePath}.zip`;
            await zip(filePath, compressedPath);
            await RNFS.unlink(filePath);
          }
        }
      }
    } catch (err) {
      console.warn('Failed to cache assets:', err);
    }
  }
  
  async getAssetPath(workflowId, assetId) {
    try {
      const workflowDir = `${RNFS.DocumentDirectoryPath}/offline/${workflowId}`;
      const files = await RNFS.readdir(workflowDir);
      
      const assetFile = files.find(file => file.includes(assetId));
      if (assetFile) {
        const filePath = `${workflowDir}/${assetFile}`;
        
        // Decompress if needed
        if (assetFile.endsWith('.zip')) {
          const extractedPath = filePath.replace('.zip', '');
          if (!(await RNFS.exists(extractedPath))) {
            await unzip(filePath, workflowDir);
          }
          return extractedPath;
        }
        
        return filePath;
      }
      
      return null;
    } catch (err) {
      console.warn('Failed to get asset path:', err);
      return null;
    }
  }
  
  async removeAssets(workflowId) {
    try {
      const workflowDir = `${RNFS.DocumentDirectoryPath}/offline/${workflowId}`;
      if (await RNFS.exists(workflowDir)) {
        await RNFS.unlink(workflowDir);
      }
    } catch (err) {
      console.warn('Failed to remove assets:', err);
    }
  }
  
  // Cache management
  async getCacheSize() {
    try {
      const offlineDir = `${RNFS.DocumentDirectoryPath}/offline`;
      if (!(await RNFS.exists(offlineDir))) {
        return 0;
      }
      
      const files = await RNFS.readdir(offlineDir);
      let totalSize = 0;
      
      for (const file of files) {
        const filePath = `${offlineDir}/${file}`;
        const stats = await RNFS.stat(filePath);
        totalSize += stats.size;
      }
      
      return totalSize;
    } catch (err) {
      console.warn('Failed to get cache size:', err);
      return 0;
    }
  }
  
  async cleanupOldCache() {
    try {
      const cacheSize = await this.getCacheSize();
      
      if (cacheSize > MAX_CACHE_SIZE) {
        // Get all cached workflows
        const cached = await this.getCachedWorkflows();
        
        // Sort by cache date (oldest first)
        const sortedWorkflows = Object.entries(cached)
          .sort(([,a], [,b]) => new Date(a.cachedAt) - new Date(b.cachedAt));
        
        // Remove oldest workflows until under limit
        let currentSize = cacheSize;
        for (const [workflowId, workflowData] of sortedWorkflows) {
          if (currentSize <= MAX_CACHE_SIZE * 0.8) break; // Leave 20% buffer
          
          await this.removeCachedWorkflow(workflowId);
          
          // Estimate size reduction (rough calculation)
          currentSize -= JSON.stringify(workflowData).length * 2;
        }
      }
    } catch (err) {
      console.warn('Failed to cleanup old cache:', err);
    }
  }
  
  async clearAllCache() {
    try {
      await AsyncStorage.multiRemove([
        STORAGE_KEYS.WORKFLOWS,
        STORAGE_KEYS.EXECUTIONS,
        STORAGE_KEYS.PENDING_SYNC
      ]);
      
      const offlineDir = `${RNFS.DocumentDirectoryPath}/offline`;
      if (await RNFS.exists(offlineDir)) {
        await RNFS.unlink(offlineDir);
        await RNFS.mkdir(offlineDir);
      }
    } catch (err) {
      console.warn('Failed to clear all cache:', err);
    }
  }
  
  // User preferences
  async saveUserPreferences(preferences) {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(preferences));
    } catch (err) {
      console.warn('Failed to save user preferences:', err);
    }
  }
  
  async getUserPreferences() {
    try {
      const prefs = await AsyncStorage.getItem(STORAGE_KEYS.USER_PREFERENCES);
      return prefs ? JSON.parse(prefs) : {};
    } catch (err) {
      console.warn('Failed to get user preferences:', err);
      return {};
    }
  }
}

export default new OfflineStorageService();
'''
			
			self.generated_files["OfflineStorageService.js"] = storage_code
			return storage_code
			
		except Exception as e:
			logger.error(f"Generate offline storage error: {e}")
			raise
	
	def generate_native_modules(self) -> Dict[str, str]:
		"""Generate native module bridges for iOS and Android"""
		try:
			native_modules = {}
			
			# iOS Native Module
			ios_module = '''
// WorkflowNativeModule.h
#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface WorkflowNativeModule : RCTEventEmitter <RCTBridgeModule>
@end

// WorkflowNativeModule.m
#import "WorkflowNativeModule.h"
#import <React/RCTLog.h>
#import <UserNotifications/UserNotifications.h>
#import <CoreHaptics/CoreHaptics.h>
#import <BackgroundTasks/BackgroundTasks.h>

@implementation WorkflowNativeModule {
  CHHapticEngine *_hapticEngine;
  NSMutableDictionary *_backgroundTasks;
}

RCT_EXPORT_MODULE();

- (instancetype)init {
  self = [super init];
  if (self) {
    _backgroundTasks = [[NSMutableDictionary alloc] init];
    
    // Initialize haptic engine
    NSError *error;
    _hapticEngine = [[CHHapticEngine alloc] initAndReturnError:&error];
    if (error) {
      RCTLogWarn(@"Failed to initialize haptic engine: %@", error.localizedDescription);
    }
  }
  return self;
}

- (NSArray<NSString *> *)supportedEvents {
  return @[@"WorkflowExecutionUpdate", @"OfflineStatusChanged", @"PushNotificationReceived"];
}

// MARK: - Haptic Feedback

RCT_EXPORT_METHOD(triggerHapticFeedback:(NSString *)type
                  intensity:(nonnull NSNumber *)intensity
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  
  if (!_hapticEngine) {
    reject(@"haptic_unavailable", @"Haptic engine not available", nil);
    return;
  }
  
  dispatch_async(dispatch_get_main_queue(), ^{
    NSError *error;
    
    if ([type isEqualToString:@"impact"]) {
      UIImpactFeedbackGenerator *generator = [[UIImpactFeedbackGenerator alloc] 
                                             initWithStyle:UIImpactFeedbackStyleMedium];
      [generator impactOccurred];
    } else if ([type isEqualToString:@"selection"]) {
      UISelectionFeedbackGenerator *generator = [[UISelectionFeedbackGenerator alloc] init];
      [generator selectionChanged];
    } else if ([type isEqualToString:@"notification"]) {
      UINotificationFeedbackGenerator *generator = [[UINotificationFeedbackGenerator alloc] init];
      [generator notificationOccurred:UINotificationFeedbackTypeSuccess];
    } else if ([type isEqualToString:@"custom"]) {
      // Custom haptic pattern
      [self playCustomHapticPattern:intensity.floatValue error:&error];
    }
    
    if (error) {
      reject(@"haptic_error", error.localizedDescription, error);
    } else {
      resolve(@YES);
    }
  });
}

- (void)playCustomHapticPattern:(float)intensity error:(NSError **)error {
  // Create custom haptic pattern for workflow feedback
  NSMutableArray *events = [[NSMutableArray alloc] init];
  
  // Sharp tap for start
  CHHapticEventParameter *intensityParam = [[CHHapticEventParameter alloc] 
                                           initWithParameterID:CHHapticEventParameterIDHapticIntensity 
                                           value:intensity];
  CHHapticEventParameter *sharpnessParam = [[CHHapticEventParameter alloc] 
                                           initWithParameterID:CHHapticEventParameterIDHapticSharpness 
                                           value:0.8];
  
  CHHapticEvent *event = [[CHHapticEvent alloc] 
                         initWithEventType:CHHapticEventTypeHapticTransient 
                         parameters:@[intensityParam, sharpnessParam] 
                         relativeTime:0];
  [events addObject:event];
  
  CHHapticPattern *pattern = [[CHHapticPattern alloc] initWithEvents:events 
                                                          parameters:@[] 
                                                               error:error];
  if (*error) return;
  
  id<CHHapticPatternPlayer> player = [_hapticEngine createPlayerWithPattern:pattern error:error];
  if (*error) return;
  
  [player startAtTime:0 error:error];
}

// MARK: - Background Tasks

RCT_EXPORT_METHOD(registerBackgroundTask:(NSString *)taskName
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  
  BGTaskRequest *request = [[BGAppRefreshTaskRequest alloc] initWithIdentifier:taskName];
  request.earliestBeginDate = [NSDate dateWithTimeIntervalSinceNow:15 * 60]; // 15 minutes
  
  NSError *error;
  BOOL success = [[BGTaskScheduler sharedScheduler] submitTaskRequest:request error:&error];
  
  if (success) {
    resolve(@YES);
  } else {
    reject(@"background_task_error", error.localizedDescription, error);
  }
}

// MARK: - Device Information

RCT_EXPORT_METHOD(getDeviceCapabilities:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  
  NSDictionary *capabilities = @{
    @"haptics": @(_hapticEngine != nil),
    @"biometrics": @([self isBiometricsAvailable]),
    @"background_processing": @YES,
    @"push_notifications": @([self isPushNotificationsEnabled]),
    @"camera": @([UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera]),
    @"location": @([CLLocationManager locationServicesEnabled]),
    @"device_model": [[UIDevice currentDevice] model],
    @"system_version": [[UIDevice currentDevice] systemVersion]
  };
  
  resolve(capabilities);
}

- (BOOL)isBiometricsAvailable {
  LAContext *context = [[LAContext alloc] init];
  NSError *error;
  return [context canEvaluatePolicy:LAPolicyDeviceOwnerAuthenticationWithBiometrics error:&error];
}

- (BOOL)isPushNotificationsEnabled {
  UNUserNotificationCenter *center = [UNUserNotificationCenter currentNotificationCenter];
  __block BOOL enabled = NO;
  
  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  [center getNotificationSettingsWithCompletionHandler:^(UNNotificationSettings *settings) {
    enabled = settings.authorizationStatus == UNAuthorizationStatusAuthorized;
    dispatch_semaphore_signal(semaphore);
  }];
  dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
  
  return enabled;
}

@end
'''
			
			# Android Native Module
			android_module = '''
// WorkflowNativeModule.java
package com.workflowapp.nativemodules;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.app.NotificationManager;

import androidx.work.WorkManager;
import androidx.work.OneTimeWorkRequest;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class WorkflowNativeModule extends ReactContextBaseJavaModule {
    
    private static final String MODULE_NAME = "WorkflowNativeModule";
    private final ReactApplicationContext reactContext;
    private Vibrator vibrator;
    
    public WorkflowNativeModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;
        this.vibrator = (Vibrator) reactContext.getSystemService(Context.VIBRATOR_SERVICE);
    }
    
    @Override
    public String getName() {
        return MODULE_NAME;
    }
    
    @Override
    public Map<String, Object> getConstants() {
        final Map<String, Object> constants = new HashMap<>();
        constants.put("HAS_VIBRATOR", vibrator != null && vibrator.hasVibrator());
        return constants;
    }
    
    // MARK: - Haptic Feedback
    
    @ReactMethod
    public void triggerHapticFeedback(String type, double intensity, Promise promise) {
        try {
            if (vibrator == null || !vibrator.hasVibrator()) {
                promise.reject("haptic_unavailable", "Vibrator not available");
                return;
            }
            
            long duration = Math.max(50, (long) (intensity * 200)); // 50-200ms based on intensity
            
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                VibrationEffect effect;
                
                switch (type) {
                    case "impact":
                        effect = VibrationEffect.createOneShot(duration, VibrationEffect.DEFAULT_AMPLITUDE);
                        break;
                    case "selection":
                        effect = VibrationEffect.createWaveform(new long[]{0, 50, 100, 50}, -1);
                        break;
                    case "notification":
                        effect = VibrationEffect.createWaveform(new long[]{0, 100, 50, 100}, -1);
                        break;
                    case "custom":
                        // Custom pattern for workflow feedback
                        long[] pattern = {0, 100, 50, 100, 50, 200};
                        effect = VibrationEffect.createWaveform(pattern, -1);
                        break;
                    default:
                        effect = VibrationEffect.createOneShot(duration, VibrationEffect.DEFAULT_AMPLITUDE);
                }
                
                vibrator.vibrate(effect);
            } else {
                // Fallback for older versions
                vibrator.vibrate(duration);
            }
            
            promise.resolve(true);
        } catch (Exception e) {
            promise.reject("haptic_error", e.getMessage());
        }
    }
    
    // MARK: - Background Tasks
    
    @ReactMethod
    public void registerBackgroundTask(String taskName, Promise promise) {
        try {
            OneTimeWorkRequest workRequest = new OneTimeWorkRequest.Builder(WorkflowSyncWorker.class)
                    .setInitialDelay(15, TimeUnit.MINUTES)
                    .addTag(taskName)
                    .build();
            
            WorkManager.getInstance(reactContext).enqueue(workRequest);
            promise.resolve(true);
        } catch (Exception e) {
            promise.reject("background_task_error", e.getMessage());
        }
    }
    
    // MARK: - Device Information
    
    @ReactMethod
    public void getDeviceCapabilities(Promise promise) {
        try {
            WritableMap capabilities = Arguments.createMap();
            
            capabilities.putBoolean("haptics", vibrator != null && vibrator.hasVibrator());
            capabilities.putBoolean("biometrics", isBiometricsAvailable());
            capabilities.putBoolean("background_processing", true);
            capabilities.putBoolean("push_notifications", isPushNotificationsEnabled());
            capabilities.putBoolean("camera", hasCamera());
            capabilities.putBoolean("location", hasLocationPermission());
            capabilities.putString("device_model", Build.MODEL);
            capabilities.putString("system_version", Build.VERSION.RELEASE);
            
            promise.resolve(capabilities);
        } catch (Exception e) {
            promise.reject("device_capabilities_error", e.getMessage());
        }
    }
    
    private boolean isBiometricsAvailable() {
        // Check for biometric authentication availability
        PackageManager pm = reactContext.getPackageManager();
        return pm.hasSystemFeature(PackageManager.FEATURE_FINGERPRINT);
    }
    
    private boolean isPushNotificationsEnabled() {
        NotificationManager notificationManager = 
            (NotificationManager) reactContext.getSystemService(Context.NOTIFICATION_SERVICE);
        return notificationManager.areNotificationsEnabled();
    }
    
    private boolean hasCamera() {
        PackageManager pm = reactContext.getPackageManager();
        return pm.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY);
    }
    
    private boolean hasLocationPermission() {
        int permission = reactContext.checkSelfPermission(android.Manifest.permission.ACCESS_FINE_LOCATION);
        return permission == PackageManager.PERMISSION_GRANTED;
    }
    
    // MARK: - Event Emission
    
    private void sendEvent(String eventName, WritableMap params) {
        reactContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
            .emit(eventName, params);
    }
    
    // Background worker for sync tasks
    public static class WorkflowSyncWorker extends Worker {
        public WorkflowSyncWorker(Context context, WorkerParameters workerParams) {
            super(context, workerParams);
        }
        
        @Override
        public Result doWork() {
            try {
                // Perform background sync
                // This would integrate with your sync service
                return Result.success();
            } catch (Exception e) {
                return Result.failure();
            }
        }
    }
}
'''
			
			native_modules["ios"] = ios_module
			native_modules["android"] = android_module
			
			self.generated_files.update({
				"ios/WorkflowNativeModule.h": ios_module.split("// WorkflowNativeModule.m")[0],
				"ios/WorkflowNativeModule.m": "// WorkflowNativeModule.m" + ios_module.split("// WorkflowNativeModule.m")[1],
				"android/WorkflowNativeModule.java": android_module
			})
			
			return native_modules
			
		except Exception as e:
			logger.error(f"Generate native modules error: {e}")
			raise
	
	def generate_package_json(self) -> str:
		"""Generate package.json for React Native project"""
		try:
			package_json = {
				"name": "APGWorkflowMobile",
				"version": "1.0.0",
				"private": True,
				"scripts": {
					"android": "react-native run-android",
					"ios": "react-native run-ios",
					"start": "react-native start",
					"test": "jest",
					"lint": "eslint . --ext .js,.jsx,.ts,.tsx",
					"build:android": "cd android && ./gradlew assembleRelease",
					"build:ios": "cd ios && xcodebuild -workspace APGWorkflowMobile.xcworkspace -scheme APGWorkflowMobile -configuration Release -destination generic/platform=iOS -archivePath APGWorkflowMobile.xcarchive archive"
				},
				"dependencies": {
					"react": "18.2.0",
					"react-native": "0.72.6",
					"@react-navigation/native": "^6.1.9",
					"@react-navigation/stack": "^6.3.20",
					"@react-navigation/bottom-tabs": "^6.5.11",
					"@reduxjs/toolkit": "^1.9.7",
					"react-redux": "^8.1.3",
					"@react-native-async-storage/async-storage": "^1.19.3",
					"@react-native-community/netinfo": "^9.4.1",
					"react-native-safe-area-context": "^4.7.4",
					"react-native-screens": "^3.25.0",
					"react-native-gesture-handler": "^2.13.4",
					"react-native-reanimated": "^3.5.4",
					"react-native-vector-icons": "^10.0.0",
					"react-native-linear-gradient": "^2.8.3",
					"react-native-haptic-feedback": "^2.2.0",
					"react-native-push-notification": "^8.1.1",
					"react-native-background-job": "^2.0.0",
					"react-native-fs": "^2.20.0",
					"react-native-zip-archive": "^6.0.7",
					"react-native-device-info": "^10.11.0",
					"react-native-keychain": "^8.1.3",
					"react-native-biometrics": "^3.0.1",
					"react-native-camera": "^4.2.1",
					"react-native-image-picker": "^5.6.0",
					"react-native-document-picker": "^9.1.1",
					"react-native-share": "^9.4.1",
					"react-native-qrcode-scanner": "^1.5.5",
					"react-native-permissions": "^3.10.1",
					"@react-native-firebase/app": "^18.6.1",
					"@react-native-firebase/messaging": "^18.6.1",
					"@react-native-firebase/analytics": "^18.6.1",
					"@react-native-firebase/crashlytics": "^18.6.1"
				},
				"devDependencies": {
					"@babel/core": "^7.20.0",
					"@babel/preset-env": "^7.20.0",
					"@babel/runtime": "^7.20.0",
					"@react-native/eslint-config": "^0.72.2",
					"@react-native/metro-config": "^0.72.11",
					"@tsconfig/react-native": "^3.0.0",
					"@types/react": "^18.0.24",
					"@types/react-test-renderer": "^18.0.0",
					"babel-jest": "^29.2.1",
					"eslint": "^8.19.0",
					"jest": "^29.2.1",
					"metro-react-native-babel-preset": "0.76.8",
					"prettier": "^2.4.1",
					"react-test-renderer": "18.2.0",
					"typescript": "4.8.4"
				},
				"jest": {
					"preset": "react-native"
				}
			}
			
			package_content = json.dumps(package_json, indent=2)
			self.generated_files["package.json"] = package_content
			return package_content
			
		except Exception as e:
			logger.error(f"Generate package.json error: {e}")
			raise
	
	def export_all_files(self) -> Dict[str, str]:
		"""Export all generated files"""
		try:
			# Generate all components if not already generated
			if not self.generated_files:
				self.generate_workflow_screen({"id": "sample", "name": "Sample Workflow"})
				self.generate_workflow_step_component()
				self.generate_workflow_hooks()
				self.generate_offline_storage_service()
				self.generate_native_modules()
				self.generate_package_json()
			
			return self.generated_files
			
		except Exception as e:
			logger.error(f"Export files error: {e}")
			raise


# Global React Native generator instance
react_native_generator = ReactNativeGenerator()