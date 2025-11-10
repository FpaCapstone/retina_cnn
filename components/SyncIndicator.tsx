import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { Cloud, CloudOff, RefreshCw, CheckCircle2 } from 'lucide-react-native';
import { useOffline } from '@/contexts/offline-context';
import Colors from '@/constants/colors';

export default function SyncIndicator() {
  const { 
    isOnline, 
    getPendingSyncCount, 
    syncPendingData, 
    isSyncing,
    lastSyncTime 
  } = useOffline();

  const pendingCount = getPendingSyncCount();
  const hasPending = pendingCount > 0;

  const formatLastSync = () => {
    if (!lastSyncTime) return 'Never';
    const now = new Date();
    const diff = now.getTime() - lastSyncTime.getTime();
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return lastSyncTime.toLocaleDateString();
  };

  if (!isOnline && !hasPending) {
    return null; // Don't show if offline and nothing to sync
  }

  return (
    <View style={styles.container}>
      {isOnline ? (
        <View style={styles.onlineContainer}>
          <Cloud size={16} color={Colors.primary.green} strokeWidth={2} />
          <Text style={styles.statusText}>Online</Text>
          
          {hasPending && (
            <>
              <View style={styles.divider} />
              <Text style={styles.pendingText}>
                {pendingCount} pending
              </Text>
              <TouchableOpacity
                onPress={() => syncPendingData()}
                disabled={isSyncing}
                style={styles.syncButton}
                activeOpacity={0.7}
              >
                {isSyncing ? (
                  <ActivityIndicator size="small" color={Colors.primary.purple} />
                ) : (
                  <RefreshCw size={14} color={Colors.primary.purple} strokeWidth={2} />
                )}
                <Text style={styles.syncButtonText}>
                  {isSyncing ? 'Syncing...' : 'Sync'}
                </Text>
              </TouchableOpacity>
            </>
          )}
          
          {!hasPending && lastSyncTime && (
            <>
              <View style={styles.divider} />
              <CheckCircle2 size={14} color={Colors.primary.green} strokeWidth={2} />
              <Text style={styles.syncedText}>
                Synced {formatLastSync()}
              </Text>
            </>
          )}
        </View>
      ) : (
        <View style={styles.offlineContainer}>
          <CloudOff size={16} color={Colors.text.secondary} strokeWidth={2} />
          <Text style={styles.offlineText}>Offline</Text>
          {hasPending && (
            <>
              <View style={styles.divider} />
              <Text style={styles.pendingText}>
                {pendingCount} will sync when online
              </Text>
            </>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: Colors.background.secondary,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border.primary,
  },
  onlineContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  offlineContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
    color: Colors.primary.green,
  },
  offlineText: {
    fontSize: 12,
    fontWeight: '600',
    color: Colors.text.secondary,
  },
  divider: {
    width: 1,
    height: 16,
    backgroundColor: Colors.border.primary,
  },
  pendingText: {
    fontSize: 12,
    color: Colors.text.secondary,
  },
  syncedText: {
    fontSize: 12,
    color: Colors.text.secondary,
  },
  syncButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
    backgroundColor: Colors.primary.purple + '20',
    borderRadius: 8,
  },
  syncButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: Colors.primary.purple,
  },
});

