"""
Resource Management Examples

Demonstrates usage of the resource management module for:
- Resource monitoring
- Progress tracking
- Automatic backup
- Report generation

Run this example:
    python examples/resource_management_examples.py
"""

import time
import logging
from typing import Dict, Any

import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Basic Resource Monitoring
# =============================================================================

def example_basic_monitoring() -> None:
    """
    Demonstrates basic resource monitoring functionality.
    
    Shows how to:
    - Initialize resource monitor
    - Get current resource metrics
    - Check thresholds
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Resource Monitoring")
    print("=" * 60 + "\n")
    
    from src.resource_management import ResourceMonitor, load_resource_config
    
    # Load configuration
    config = load_resource_config()
    
    # Create monitor
    monitor = ResourceMonitor(config)
    
    # Get current metrics
    metrics = monitor.get_all_metrics()
    
    print("Current Resource Usage:")
    print("-" * 40)
    
    if metrics.cpu:
        print(f"  CPU: {metrics.cpu.usage_percent:.1f}%")
    
    if metrics.memory:
        info = metrics.memory.additional_info
        print(f"  Memory: {metrics.memory.usage_percent:.1f}%")
        print(f"    Used: {info.get('used_mb', 0):.0f} MB")
        print(f"    Available: {info.get('available_mb', 0):.0f} MB")
    
    if metrics.gpu and metrics.gpu.additional_info.get("available", True):
        print(f"  GPU Memory: {metrics.gpu.usage_percent:.1f}%")
        print(f"    Device: {metrics.gpu.additional_info.get('device_name', 'Unknown')}")
    
    if metrics.disk:
        info = metrics.disk.additional_info
        print(f"  Disk: {metrics.disk.usage_percent:.1f}%")
        print(f"    Used: {info.get('used_gb', 0):.1f} GB")
        print(f"    Free: {info.get('free_gb', 0):.1f} GB")
    
    if metrics.alerts:
        print("\nActive Alerts:")
        for alert in metrics.alerts:
            print(f"  - {alert}")
    else:
        print("\nNo active alerts.")


# =============================================================================
# Example 2: Progress Tracking
# =============================================================================

def example_progress_tracking() -> None:
    """
    Demonstrates progress tracking functionality.
    
    Shows how to:
    - Create and manage tasks
    - Update progress
    - Track training epochs
    """
    print("\n" + "=" * 60)
    print("Example 2: Progress Tracking")
    print("=" * 60 + "\n")
    
    from src.resource_management import TrainingProgressTracker, TaskStatus
    
    # Create tracker
    tracker = TrainingProgressTracker()
    
    # Create training task
    task_id = tracker.create_training_task(
        name="Example Training",
        total_epochs=3,
        steps_per_epoch=10,
    )
    
    print(f"Created task: {task_id}")
    
    # Start task
    tracker.start_task(task_id)
    
    # Simulate training
    global_step = 0
    for epoch in range(3):
        tracker.start_epoch(task_id, epoch, learning_rate=0.001)
        print(f"\nEpoch {epoch + 1}/3")
        
        epoch_losses = []
        for step in range(10):
            global_step += 1
            
            # Simulate loss
            loss = 1.0 / (global_step + 1)
            epoch_losses.append(loss)
            
            # Update progress
            task = tracker.update_training_step(
                task_id=task_id,
                step=global_step,
                loss=loss,
            )
            
            if step % 5 == 0:
                print(f"  Step {step + 1}/10 - Loss: {loss:.4f} - Progress: {task.progress_percent:.1f}%")
        
        # End epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        tracker.end_epoch(
            task_id=task_id,
            epoch=epoch,
            train_loss=avg_loss,
            val_loss=avg_loss * 0.9,  # Simulated validation loss
            metrics={"accuracy": 0.85 + epoch * 0.05},
        )
        print(f"  Epoch complete - Avg Loss: {avg_loss:.4f}")
    
    # Complete task
    task = tracker.complete_task(task_id)
    print(f"\nTask completed: {task.status.value}")
    
    # Get summary
    summary = tracker.get_training_summary(task_id)
    print("\nTraining Summary:")
    print(f"  Total epochs: {summary['total_epochs']}")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Best metrics: {summary['best_metrics']}")


# =============================================================================
# Example 3: Resource Manager with Training
# =============================================================================

def example_resource_managed_training() -> None:
    """
    Demonstrates integrated resource management during training.
    
    Shows how to:
    - Use ResourceManager for training
    - Register model for backup
    - Handle progress tracking with monitoring
    """
    print("\n" + "=" * 60)
    print("Example 3: Resource Managed Training")
    print("=" * 60 + "\n")
    
    from src.resource_management import (
        ResourceManager,
        resource_managed_training,
    )
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Using context manager for resource-managed training...")
    
    # Use context manager
    with resource_managed_training(model, optimizer) as manager:
        print(f"Manager running: {manager.is_running}")
        
        # Start training task
        task_id = manager.start_training(
            name="Example Model Training",
            total_epochs=2,
            steps_per_epoch=5,
        )
        
        # Simulate training
        global_step = 0
        for epoch in range(2):
            manager.start_epoch(epoch=epoch)
            
            for step in range(5):
                global_step += 1
                
                # Simulate forward pass
                x = torch.randn(4, 10)
                y = model(x)
                loss = y.mean()
                
                # Update progress
                manager.update_training_step(
                    step=global_step,
                    loss=loss.item(),
                )
                
                time.sleep(0.1)  # Simulate training time
            
            manager.end_epoch(
                epoch=epoch,
                train_loss=0.5 - epoch * 0.1,
            )
            print(f"  Completed epoch {epoch + 1}")
        
        # Complete training
        manager.complete_training(task_id)
        
        # Get status
        status = manager.get_status_summary()
        print(f"\nFinal Status:")
        print(f"  Is training: {status['training']['is_training']}")
        print(f"  Active alerts: {status['resources']['active_alerts']}")
    
    print("\nTraining completed with resource management.")


# =============================================================================
# Example 4: Backup and Restore
# =============================================================================

def example_backup_restore() -> None:
    """
    Demonstrates backup and restore functionality.
    
    Shows how to:
    - Register model for backup
    - Create manual backup
    - Restore from backup
    """
    print("\n" + "=" * 60)
    print("Example 4: Backup and Restore")
    print("=" * 60 + "\n")
    
    from src.resource_management import (
        BackupHandler,
        BackupType,
        BackupState,
    )
    
    # Create model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Initialize backup handler
    handler = BackupHandler()
    handler.register_model(model, optimizer)
    
    print("Creating backup...")
    
    # Create backup
    backup_info = handler.create_backup(
        backup_type=BackupType.MANUAL,
        trigger_reason="Example backup",
        additional_state={
            "epoch": 5,
            "best_accuracy": 0.92,
        },
    )
    
    print(f"  Backup ID: {backup_info.backup_id}")
    print(f"  Status: {backup_info.status.value}")
    print(f"  Path: {backup_info.filepath}")
    print(f"  Size: {backup_info.size_bytes / 1024:.1f} KB")
    
    if backup_info.filepath:
        # Modify model weights
        original_weight = model.weight.data.clone()
        model.weight.data.fill_(0)
        print(f"\nModified model weight sum: {model.weight.data.sum().item()}")
        
        # Restore from backup
        print("\nRestoring from backup...")
        restored = handler.restore_from_backup(backup_info.filepath)
        
        print(f"  Restored epoch: {restored.get('epoch', 'N/A')}")
        print(f"  Restored model weight sum: {model.weight.data.sum().item():.4f}")
        print(f"  Original weight sum: {original_weight.sum().item():.4f}")
        
        # Cleanup
        import os
        if os.path.exists(backup_info.filepath):
            os.remove(backup_info.filepath)
            print("\n  Cleaned up backup file.")


# =============================================================================
# Example 5: Report Generation
# =============================================================================

def example_report_generation() -> None:
    """
    Demonstrates report generation functionality.
    
    Shows how to:
    - Generate different report types
    - Save reports in various formats
    """
    print("\n" + "=" * 60)
    print("Example 5: Report Generation")
    print("=" * 60 + "\n")
    
    from src.resource_management import (
        ReportManager,
        ReportGenerator,
        ReportType,
        ReportFormat,
        ResourceMonitor,
        load_resource_config,
    )
    
    # Create components
    config = load_resource_config()
    monitor = ResourceMonitor(config)
    generator = ReportGenerator()
    
    # Get current metrics
    metrics = monitor.get_all_metrics()
    
    # Generate resource report
    report = generator.generate_resource_report(metrics)
    
    print(f"Generated Report:")
    print(f"  ID: {report.report_id}")
    print(f"  Type: {report.report_type.value}")
    print(f"  Title: {report.title}")
    print(f"  Summary: {report.summary}")
    
    # Initialize report manager and save
    manager = ReportManager()
    
    # Save in different formats
    print("\nSaving reports in different formats:")
    
    for fmt in [ReportFormat.JSON, ReportFormat.YAML, ReportFormat.TEXT]:
        path = manager.save_report(report, fmt)
        print(f"  {fmt.value}: {path}")
    
    # Get metadata
    metadata = manager.get_metadata()
    print(f"\nReport Metadata:")
    print(f"  Total reports: {metadata.total_reports}")
    print(f"  Total size: {metadata.total_size_bytes / 1024:.1f} KB")


# =============================================================================
# Example 6: Full Integration
# =============================================================================

def example_full_integration() -> None:
    """
    Demonstrates full integration of all components.
    
    Shows a complete workflow combining:
    - Resource monitoring
    - Progress tracking
    - Automatic backup
    - Report generation
    """
    print("\n" + "=" * 60)
    print("Example 6: Full Integration")
    print("=" * 60 + "\n")
    
    from src.resource_management import (
        ResourceManager,
        get_resource_manager,
        reset_resource_manager,
    )
    
    # Reset any existing manager
    reset_resource_manager()
    
    # Get global manager
    manager = get_resource_manager()
    
    # Create model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )
    optimizer = torch.optim.Adam(model.parameters())
    
    try:
        # Start manager
        manager.start()
        print("Resource manager started.")
        
        # Register model
        manager.register_model(model, optimizer)
        print("Model registered for backup.")
        
        # Check initial resources
        metrics = manager.get_current_metrics()
        print(f"\nInitial Resource Usage:")
        print(f"  CPU: {metrics.cpu.usage_percent:.1f}%")
        print(f"  Memory: {metrics.memory.usage_percent:.1f}%")
        
        # Start training
        task_id = manager.start_training(
            name="Full Integration Demo",
            total_epochs=2,
            steps_per_epoch=3,
        )
        print(f"\nStarted training: {task_id}")
        
        # Training loop
        global_step = 0
        for epoch in range(2):
            manager.start_epoch(epoch=epoch, learning_rate=0.001)
            
            for step in range(3):
                global_step += 1
                
                # Simulate training
                x = torch.randn(8, 100)
                y = model(x)
                loss = y.sum()
                
                manager.update_training_step(
                    step=global_step,
                    loss=loss.item(),
                    metrics={"batch_size": 8},
                )
                
                time.sleep(0.05)
            
            manager.end_epoch(
                epoch=epoch,
                train_loss=0.5,
                val_loss=0.45,
                metrics={"accuracy": 0.8},
            )
            print(f"  Epoch {epoch + 1} completed")
        
        # Complete training
        manager.complete_training(task_id)
        print("\nTraining completed.")
        
        # Generate final report
        report_path = manager.generate_report()
        print(f"Final report saved: {report_path}")
        
        # Get status summary
        status = manager.get_status_summary()
        print(f"\nFinal Status:")
        print(f"  Total backups: {status['backup']['total_backups']}")
        print(f"  Active tasks: {status['training']['active_tasks']}")
        
    finally:
        # Stop manager
        manager.stop()
        print("\nResource manager stopped.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("RESOURCE MANAGEMENT MODULE EXAMPLES")
    print("=" * 70)
    
    examples = [
        ("Basic Monitoring", example_basic_monitoring),
        ("Progress Tracking", example_progress_tracking),
        ("Resource Managed Training", example_resource_managed_training),
        ("Backup and Restore", example_backup_restore),
        ("Report Generation", example_report_generation),
        ("Full Integration", example_full_integration),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
