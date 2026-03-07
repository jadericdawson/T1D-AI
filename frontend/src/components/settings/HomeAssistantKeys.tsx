/**
 * Home Assistant API Key Management Component
 * Allows users to generate, view, and revoke API keys for HA integration.
 */
import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Key, Plus, Trash2, Copy, Check, Loader2, ExternalLink } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@/components/ui/card'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { homeAssistantApi, ApiKey } from '@/lib/api'

export function HomeAssistantKeys() {
  const queryClient = useQueryClient()
  const [newKeyName, setNewKeyName] = useState('')
  const [createdKey, setCreatedKey] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const { data: keys = [], isLoading } = useQuery({
    queryKey: ['api-keys'],
    queryFn: homeAssistantApi.listApiKeys,
  })

  const createMutation = useMutation({
    mutationFn: (name: string) => homeAssistantApi.createApiKey(name),
    onSuccess: (data) => {
      setCreatedKey(data.key || null)
      setNewKeyName('')
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
  })

  const revokeMutation = useMutation({
    mutationFn: (keyId: string) => homeAssistantApi.revokeApiKey(keyId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
  })

  const handleCopy = async (text: string) => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleCreate = () => {
    if (!newKeyName.trim()) return
    setCreatedKey(null)
    createMutation.mutate(newKeyName.trim())
  }

  const baseUrl = window.location.origin

  const yamlExample = `# Home Assistant configuration.yaml
rest:
  - resource: "${baseUrl}/api/v1/ha/status"
    headers:
      X-API-Key: "YOUR_API_KEY_HERE"
    scan_interval: 300
    sensor:
      - name: "T1D Blood Sugar"
        value_template: "{{ value_json.glucose }}"
        unit_of_measurement: "mg/dL"
        json_attributes:
          - trend_arrow
          - iob
          - cob
          - status
          - minutes_ago
      - name: "T1D Status"
        value_template: "{{ value_json.status }}"
      - name: "T1D IOB"
        value_template: "{{ value_json.iob }}"
        unit_of_measurement: "U"
      - name: "T1D COB"
        value_template: "{{ value_json.cob }}"
        unit_of_measurement: "g"

# Example automation for high BG alert
automation:
  - alias: "T1D High Blood Sugar Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.t1d_blood_sugar
        above: 250
        for: "00:15:00"
    action:
      - service: notify.notify
        data:
          title: "High Blood Sugar Alert"
          message: "BG is {{ states('sensor.t1d_blood_sugar') }} mg/dL ({{ state_attr('sensor.t1d_blood_sugar', 'trend_arrow') }})"

  - alias: "T1D Low Blood Sugar Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.t1d_blood_sugar
        below: 70
    action:
      - service: notify.notify
        data:
          title: "LOW Blood Sugar Alert!"
          message: "BG is {{ states('sensor.t1d_blood_sugar') }} mg/dL - treat immediately"`

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="w-5 h-5" />
            Home Assistant Integration
          </CardTitle>
          <CardDescription>
            Connect your Home Assistant to get real-time glucose alerts, automations, and dashboard sensors.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Create new key */}
          <div className="space-y-3">
            <Label>Create API Key</Label>
            <div className="flex gap-2">
              <Input
                placeholder="Key name (e.g., Home Assistant)"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
              />
              <Button
                onClick={handleCreate}
                disabled={!newKeyName.trim() || createMutation.isPending}
              >
                {createMutation.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Plus className="w-4 h-4" />
                )}
                <span className="ml-1">Create</span>
              </Button>
            </div>
          </div>

          {/* Show newly created key */}
          {createdKey && (
            <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/30 space-y-2">
              <p className="text-sm font-medium text-green-600 dark:text-green-400">
                API key created! Copy it now — it won't be shown again.
              </p>
              <div className="flex gap-2">
                <code className="flex-1 p-2 rounded bg-background font-mono text-sm break-all">
                  {createdKey}
                </code>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleCopy(createdKey)}
                >
                  {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </Button>
              </div>
            </div>
          )}

          <Separator />

          {/* Existing keys */}
          <div className="space-y-3">
            <Label>Active API Keys</Label>
            {isLoading ? (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading...
              </div>
            ) : keys.length === 0 ? (
              <p className="text-sm text-muted-foreground">No API keys yet. Create one above.</p>
            ) : (
              <div className="space-y-2">
                {keys.map((key: ApiKey) => (
                  <div
                    key={key.id}
                    className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                  >
                    <div>
                      <p className="font-medium">{key.name}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <code className="text-xs text-muted-foreground">{key.key_prefix}...</code>
                        {key.last_used_at && (
                          <span className="text-xs text-muted-foreground">
                            Last used: {new Date(key.last_used_at).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </div>
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button variant="ghost" size="sm" className="text-destructive">
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Revoke API Key?</AlertDialogTitle>
                          <AlertDialogDescription>
                            This will immediately disconnect any Home Assistant using this key.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() => revokeMutation.mutate(key.id)}
                            className="bg-destructive text-destructive-foreground"
                          >
                            Revoke
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </div>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Setup guide */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Setup Guide</CardTitle>
          <CardDescription>
            Add this to your Home Assistant configuration.yaml to get glucose sensors and alerts.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="relative">
            <pre className="p-4 rounded-lg bg-muted text-xs overflow-x-auto whitespace-pre max-h-96">
              {yamlExample}
            </pre>
            <Button
              variant="outline"
              size="sm"
              className="absolute top-2 right-2"
              onClick={() => handleCopy(yamlExample)}
            >
              {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              <span className="ml-1 text-xs">Copy</span>
            </Button>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ExternalLink className="w-4 h-4" />
            <span>
              Replace YOUR_API_KEY_HERE with the key you generated above.
              Restart Home Assistant after saving configuration.yaml.
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
