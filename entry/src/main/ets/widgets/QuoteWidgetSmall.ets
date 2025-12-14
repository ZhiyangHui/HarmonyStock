const storage = new LocalStorage()

@Entry(storage)
@Component
struct QuoteWidgetSmall {
  @LocalStorageProp('formId') formId: string = ''
  @LocalStorageProp('title') title: string = ''
  @LocalStorageProp('code') code: string = ''
  @LocalStorageProp('name') name: string = ''
  @LocalStorageProp('priceText') priceText: string = '--'
  @LocalStorageProp('changeText') changeText: string = '--'
  @LocalStorageProp('pctText') pctText: string = '--'
  @LocalStorageProp('tsText') tsText: string = ''

  private isUp(): boolean {
    const t: string = this.changeText.trim()
    if (t.length === 0 || t === '--') return true
    return !t.startsWith('-')
  }

  build() {
    Column() {
      Row() {
        Text(this.title.length > 0 ? this.title : '行情')
          .fontSize(10)
          .fontColor('#BBBBBB')

        Blank()

        Text(this.tsText)
          .fontSize(10)
          .fontColor('#666666')
      }
      .width('100%')

      Text(this.name.length > 0 ? this.name : '—')
        .fontSize(14)
        .fontColor('#FFFFFF')
        .margin({ top: 6 })

      Text(this.code)
        .fontSize(10)
        .fontColor('#666666')
        .margin({ top: 2 })

      Text(this.priceText)
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor(this.isUp() ? '#FF4D4F' : '#39D98A')
        .margin({ top: 8 })

      Text(`${this.changeText}  ${this.pctText}`)
        .fontSize(11)
        .fontColor(this.isUp() ? '#FF7875' : '#6BE49B')
        .margin({ top: 4 })
    }
    .padding(12)
    .width('100%')
    .height('100%')
    .backgroundColor('#151515')
    .borderRadius(16)
  }
}
