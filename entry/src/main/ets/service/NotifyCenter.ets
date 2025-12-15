import notificationManager from '@ohos.notificationManager'
import notification from '@ohos.notification'
import { hilog } from '@kit.PerformanceAnalysisKit'

const DOMAIN: number = 0x0000
const TAG: string = 'HarmonyStock.Notify'

export interface QuoteAlertPayload {
  id: number
  title: string
  text: string
  additionalText: string
}

export class NotifyCenter {
  static async ensureEnabled(): Promise<boolean> {
    try {
      const enabled: boolean = await notificationManager.isNotificationEnabled()
      if (enabled) return true
      await notificationManager.requestEnableNotification()
      return await notificationManager.isNotificationEnabled()
    } catch (e) {
      hilog.error(DOMAIN, TAG, 'ensureEnabled err=%{public}s', JSON.stringify(e))
      return false
    }
  }

  static async publish(p: QuoteAlertPayload): Promise<void> {
    const ok: boolean = await NotifyCenter.ensureEnabled()
    if (!ok) return

    const req: notificationManager.NotificationRequest = {
      id: p.id,
      // ✅ 关键：指定系统内置 SlotType
      slotType: notification.SlotType.SOCIAL_COMMUNICATION,
      content: {
        notificationContentType: notificationManager.ContentType.NOTIFICATION_CONTENT_BASIC_TEXT,
        normal: {
          title: p.title,
          text: p.text,
          additionalText: p.additionalText
        }
      }
    }

    try {
      await notificationManager.publish(req)
      hilog.info(DOMAIN, TAG, 'publish ok id=%{public}d', req.id)
    } catch (e) {
      const errObj: Record<string, Object> = e as Record<string, Object>
      const codeNum: number = Number(errObj['code'])
      hilog.error(DOMAIN, TAG, 'publish failed code=%{public}d err=%{public}s', codeNum, JSON.stringify(e))
    }
  }
}
