import notificationManager from '@ohos.notificationManager'
import notification from '@ohos.notification'
import { hilog } from '@kit.PerformanceAnalysisKit'
import { Haptics } from '../common/Haptics'
import wantAgent from '@ohos.wantAgent'  // 正确导入 wantAgent 模块

const DOMAIN: number = 0x0000
const TAG: string = 'HarmonyStock.Notify'

export interface QuoteAlertPayload {
  id: number
  title: string
  text: string
  additionalText: string
  bundleName: string  // 目标应用包名
  abilityName: string  // 目标应用的 Ability 名称
}

// 定义一个 ErrorObject 接口来避免使用字面量作为类型
interface ErrorObject {
  code: string
}

// 修改接口名称为更具描述性的名称
interface WantAgentRequest {
  bundleName: string;
  abilityName: string;
}

interface WantAgentConfig {
  wants: Array<WantAgentRequest>  // 明确声明 wants 数组的类型
  operationType: wantAgent.OperationType
  requestCode: number
  wantAgentFlags: Array<wantAgent.WantAgentFlags>
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
    const ok = await NotifyCenter.ensureEnabled()
    if (!ok) return

    try {
      // 创建 WantAgent 配置对象，指定类型
      const wantAgentInfo: WantAgentConfig = {
        wants: [{
          bundleName: p.bundleName,  // 设置目标应用包名
          abilityName: p.abilityName, // 设置目标应用 Ability 名称
        }],
        operationType: wantAgent.OperationType.START_ABILITY, // 打开 Ability
        requestCode: p.id,
        wantAgentFlags: [wantAgent.WantAgentFlags.UPDATE_PRESENT_FLAG], // 更新展示状态
      }

      // 获取 WantAgent 实例
      const agent = await wantAgent.getWantAgent(wantAgentInfo)

      const req: notificationManager.NotificationRequest = {
        id: p.id,
        slotType: notification.SlotType.SOCIAL_COMMUNICATION,
        content: {
          notificationContentType: notificationManager.ContentType.NOTIFICATION_CONTENT_BASIC_TEXT,
          normal: {
            title: p.title,
            text: p.text,
            additionalText: p.additionalText,
          },
        },
        wantAgent: agent, // 将 WantAgent 绑定到通知中
      }

      // 发布通知
      await notificationManager.publish(req)
      hilog.info(DOMAIN, TAG, 'publish ok id=%{public}d', req.id)

      // 震动反馈
      await Haptics.vibrateShort()
    } catch (e) {
      // 使用 ErrorObject 类型来处理错误
      const errObj = e as ErrorObject
      const codeNum = Number(errObj.code) // 获取错误代码
      hilog.error(
        DOMAIN,
        TAG,
        'publish failed code=%{public}d err=%{public}s',
        codeNum,
        JSON.stringify(e),
      )
    }
  }
}
